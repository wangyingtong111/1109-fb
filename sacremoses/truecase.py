# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import re
from collections import defaultdict, Counter
from functools import partial
from itertools import chain

from sacremoses.corpus import Perluniprops
from sacremoses.util import parallelize_preprocess, grouper


perluniprops = Perluniprops()


#: Header written as the first line of every model file that records the true
#: count of *every* surface variant. Count-bearing models can be merged by
#: adding their per-variant counts; the original Moses/Perl format cannot,
#: because it writes a placeholder ``(1)`` for every non-best variant, so the
#: real counts are gone the moment the file is written. The header is the
#: unambiguous marker that distinguishes the two: a bare ``(1)`` is otherwise
#: indistinguishable from a variant that genuinely occurred once.
MODEL_HEADER = "# sacremoses truecase model v2"


class LegacyModelError(ValueError):
    """Raised when a model without recoverable per-variant counts is asked to
    take part in a merge.

    Files in the original Moses/Perl layout (and every model written before the
    v2 header existed) stay *readable* for truecasing -- ``MosesTruecaser(path)``
    still loads them. They only cannot be merged, because a non-best variant is
    stored as ``(1)`` regardless of its real frequency, so its count cannot be
    recovered. Re-train such a shard from its corpus with this version, or
    merge the underlying corpora and train once.
    """


class MosesTruecaser(object):
    """
    This is a Python port of the Moses Truecaser from
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/train-truecaser.perl
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/truecase.perl
    """

    #: Upper bound on distinct lowercased types read from a model file. A
    #: model is often downloaded or shared, and loading amplifies its size
    #: about 35x in memory, so this is a backstop against a hostile one -- not
    #: a limit on legitimate vocabularies, which sit far below it.
    MAX_MODEL_ENTRIES = 2_000_000

    # Perl Unicode Properties character sets.
    Lowercase_Letter = str("".join(perluniprops.chars("Lowercase_Letter")))
    Uppercase_Letter = str("".join(perluniprops.chars("Uppercase_Letter")))
    Titlecase_Letter = str("".join(perluniprops.chars("Uppercase_Letter")))

    def __init__(self, load_from=None, is_asr=None, encoding="utf8"):
        """
        :param load_from:
        :type load_from:

        :param is_asr: A flag to indicate that model is for ASR. ASR input has
            no case, make sure it is lowercase, and make sure known are cased
            eg. 'i' to be uppercased even if i is known.
        :type is_asr: bool
        """
        # Initialize the object.
        super(MosesTruecaser, self).__init__()
        # Initialize the language specific nonbreaking prefixes.
        self.SKIP_LETTERS_REGEX = re.compile(
            "[{}{}{}]".format(
                self.Lowercase_Letter, self.Uppercase_Letter, self.Titlecase_Letter
            )
        )

        self.XML_SPLIT_REGX = re.compile("(<.*(?<=>))(.*)((?=</)[^>]*>)")

        self.SENT_END = {".", ":", "?", "!"}
        self.DELAYED_SENT_START = {
            "(",
            "[",
            '"',
            "'",
            "&apos;",
            "&quot;",
            "&#91;",
            "&#93;",
        }

        self.encoding = encoding

        self.is_asr = is_asr
        if load_from:
            self.model = self._load_model(load_from)

    def learn_truecase_weights(self, tokens, possibly_use_first_token=False):
        """
        This function checks through each tokens in a sentence and returns the
        appropriate weight of each surface token form.
        """
        # Keep track of first tokens in the sentence(s) of the line.
        is_first_word = True
        truecase_weights = []
        for i, token in enumerate(tokens):
            # Skip XML tags.
            if re.search(r"(<\S[^>]*>)", token):
                continue
            # Skip if sentence start symbols.
            elif token in self.DELAYED_SENT_START:
                continue

            # Resets the `is_first_word` after seeing sent end symbols.
            if not is_first_word and token in self.SENT_END:
                is_first_word = True
                continue
            # Skips tokens with nothing to case.
            if not self.SKIP_LETTERS_REGEX.search(token):
                is_first_word = False
                continue

            # If it's not the first word,
            # then set the current word weight to 1.
            current_word_weight = 0
            if not is_first_word:
                current_word_weight = 1
            # Otherwise check whether user wants to optionally
            # use the first word.
            elif possibly_use_first_token:
                # Gated special handling of first word of sentence.
                # Check if first characer of token is lowercase.
                if token[0].islower():
                    current_word_weight = 1
                elif i == 1:
                    current_word_weight = 0.1

            is_first_word = False

            if current_word_weight > 0:
                truecase_weights.append((token.lower(), token, current_word_weight))
        return truecase_weights

    def _train(
        self,
        document_iterator,
        save_to=None,
        possibly_use_first_token=False,
        processes=1,
        progress_bar=False,
    ):
        """
        :param document_iterator: The input document, each outer list is a sentence,
                          the inner list is the list of tokens for each sentence.
        :type document_iterator: iter(list(str))

        :param possibly_use_first_token: When True, on the basis that the first
            word of a sentence is always capitalized; if this option is provided then:
            a) if a sentence-initial token is *not* capitalized, then it is counted, and
            b) if a capitalized sentence-initial token is the only token of the segment,
               then it is counted, but with only 10% of the weight of a normal token.
        :type possibly_use_first_token: bool

        :returns: A dictionary of the best, known objects as values from `_casing_to_model()`
        :rtype: {'best': dict, 'known': Counter}
        """
        casing = defaultdict(Counter)
        train_truecaser = partial(
            self.learn_truecase_weights,
            possibly_use_first_token=possibly_use_first_token,
        )
        token_weights = chain(
            *parallelize_preprocess(
                train_truecaser, document_iterator, processes, progress_bar=progress_bar
            )
        )
        # Collect the token_weights from every sentence.
        for lowercase_token, surface_token, weight in token_weights:
            casing[lowercase_token][surface_token] += weight

        # Save to file if specified.
        if save_to:
            self._save_model_from_casing(casing, save_to)
        return self._casing_to_model(casing)

    def train(
        self,
        documents,
        save_to=None,
        possibly_use_first_token=False,
        processes=1,
        progress_bar=False,
    ):
        """
        Default duck-type of _train(), accepts list(list(str)) as input documents.
        """
        self.model = None  # Clear the model first.
        self.model = self._train(
            documents,
            save_to,
            possibly_use_first_token,
            processes,
            progress_bar=progress_bar,
        )
        return self.model

    def train_from_file(
        self,
        filename,
        save_to=None,
        possibly_use_first_token=False,
        processes=1,
        progress_bar=False,
    ):
        """
        Duck-type of _train(), accepts a filename to read as a `iter(list(str))`
        object.
        """
        with open(filename, encoding=self.encoding) as fin:
            # document_iterator = map(str.split, fin.readlines())
            document_iterator = (
                line.split() for line in fin.readlines()
            )  # Lets try a generator comprehension for Python2...
        self.model = None  # Clear the model first.
        self.model = self._train(
            document_iterator,
            save_to,
            possibly_use_first_token,
            processes,
            progress_bar=progress_bar,
        )
        return self.model

    def train_from_file_object(
        self,
        file_object,
        save_to=None,
        possibly_use_first_token=False,
        processes=1,
        progress_bar=False,
    ):
        """
        Duck-type of _train(), accepts a file object to read as a `iter(list(str))`
        object.
        """
        # document_iterator = map(str.split, file_object.readlines())
        document_iterator = (
            line.split() for line in file_object.readlines()
        )  # Lets try a generator comprehension for Python2...
        self.model = None  # Clear the model first.
        self.model = self._train(
            document_iterator,
            save_to,
            possibly_use_first_token,
            processes,
            progress_bar=progress_bar,
        )
        return self.model

    def truecase(self, text, return_str=False, use_known=False):
        """
        Truecase a single sentence / line of text.

        :param text: A single string, i.e. sentence text.
        :type text: str

        :param use_known: Use the known case if a word is a known word but not the first word.
        :type use_known: bool
        """
        check_model_message = str(
            "\nUse Truecaser.train() to train a model.\n"
            "Or use Truecaser('modefile') to load a model."
        )
        # Not an assert: `python -O` strips it, and the caller would then get
        # an opaque AttributeError deep inside the loop instead of this message.
        if not hasattr(self, "model"):
            raise ValueError(check_model_message)
        # Keep track of first tokens in the sentence(s) of the line.
        is_first_word = True
        truecased_tokens = []
        tokens = self.split_xml(text)
        # best_cases = best_cases if best_cases else self.model['best']
        # known_cases = known_cases if known_cases else self.model['known']

        for i, token in enumerate(tokens):

            # Append XML tags and continue
            if re.search(r"(<\S[^>]*>)", token):
                truecased_tokens.append(token)
                continue

            # Note this shouldn't happen other if | are escaped as &#124;
            # To make the truecaser resilient,
            # we'll just any token starting with pipes as they are.
            if token == "|" or token.startswith("|"):
                truecased_tokens.append(token)
                continue

            # Reads the word token and factors separatedly
            token, other_factors = re.search(r"^([^\|]+)(.*)", token).groups()

            # Lowercase the ASR tokens.
            if self.is_asr:
                token = token.lower()

            # The actual case replacement happens here.
            # "Most frequent" case of the word.
            best_case = self.model["best"].get(token.lower(), None)
            # If it's the start of sentence.
            if is_first_word and best_case:  # Truecase sentence start.
                token = best_case
            elif use_known and token in self.model["known"]:  # Don't change known tokens.
                pass
            elif best_case:  # Truecase otherwise unknown tokens? Heh? From https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/truecase.perl#L66
                token = best_case
            # Else, it's an unknown word, don't change the word.
            # Concat the truecased `word` with the `other_factors`
            token = token + other_factors
            # Adds the truecased word.
            truecased_tokens.append(token)

            # Resets sentence start if this token is an ending punctuation.
            if token in self.SENT_END:
                is_first_word = True
            elif token not in self.DELAYED_SENT_START:
                is_first_word = False

        # return ' '.join(tokens)
        return " ".join(truecased_tokens) if return_str else truecased_tokens

    def truecase_file(self, filename, return_str=True):
        with open(filename, encoding=self.encoding) as fin:
            for line in fin:
                truecased_tokens = self.truecase(line.strip())
                # Yield the truecased line.
                yield " ".join(truecased_tokens) if return_str else truecased_tokens

    @staticmethod
    def split_xml(line):
        """
        Python port of split_xml function in Moses' truecaser:
        https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/truecaser.perl

        :param line: Input string, should be tokenized, separated by space.
        :type line: str
        """
        line = line.strip()
        tokens = []
        while line:
            # Assumes that xml tag is always separated by space.
            has_xml = re.search(r"^\s*(<\S[^>]*>)(.*)$", line)
            # non-XML test.
            is_non_xml = re.search(r"^\s*([^\s<>]+)(.*)$", line)
            # '<' or '>' occurs in word, but it's not an XML tag
            xml_cognates = re.search(r"^\s*(\S+)(.*)$", line)
            if has_xml:
                potential_xml, line_next = has_xml.groups()
                # exception for factor that is an XML tag
                if (
                    re.search(r"^\S", line)
                    and len(tokens) > 0
                    and re.search(r"\|$", tokens[-1])
                ):
                    tokens[-1] += potential_xml
                    # If it's a token with factors, join with the previous token.
                    is_factor = re.search(r"^(\|+)(.*)$", line_next)
                    if is_factor:
                        tokens[-1] += is_factor.group(1)
                        line_next = is_factor.group(2)
                else:
                    tokens.append(
                        potential_xml + " "
                    )  # Token hack, unique to sacremoses.
                line = line_next

            elif is_non_xml:
                tokens.append(is_non_xml.group(1))  # Token hack, unique to sacremoses.
                line = is_non_xml.group(2)
            elif xml_cognates:
                tokens.append(
                    xml_cognates.group(1)
                )  # Token hack, unique to sacremoses.
                line = xml_cognates.group(2)
            else:
                raise Exception("ERROR: huh? {}".format(line))
            tokens[-1] = tokens[-1].strip()  # Token hack, unique to sacremoses.
        return tokens

    @staticmethod
    def _ranked_variants(variant_counts):
        """The surface variants of one lowercased token, in a total order.

        Primary key is the true count, most frequent first. Ties are broken by
        the surface form in ascending lexicographic order, so the result never
        depends on dict/insertion order. That makes ``_casing_to_model`` -- and
        every merge built on top of it -- independent of how the corpus was
        shuffled, split into batches, or the order in which the batches were
        merged.
        """
        return sorted(
            variant_counts.items(),
            key=lambda surface_count: (-surface_count[1], surface_count[0]),
        )

    def _casing_to_model(self, casing):
        """

        :returns: A tuple of the (best, known) objects.
        :rtype: tuple(dict, Counter)
        """
        best = {}
        known = Counter()

        for token_lower in casing:
            tokens = self._ranked_variants(casing[token_lower])
            # Set the most frequent case as the "best" case. On an exact count
            # tie the lexicographically smallest surface form wins, deterministically.
            best[token_lower] = tokens[0][0]
            # If it's asr, throw away everything
            if not self.is_asr:
                for token, count in tokens[1:]:
                    # Note: This is rather odd that the counts are thrown away...
                    # from https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/truecase.perl#L34
                    known[token] += 1
        model = {"best": best, "known": known, "casing": casing}
        return model

    def save_model(self, filename):
        self._save_model_from_casing(self.model["casing"], filename)

    def _save_model_from_casing(self, casing, filename):
        """
        Outputs the truecaser model file in the same output format as
        https://github.com/moses-smt/mosesdecoder/blob/master/scripts/recaser/train-truecaser.perl

        :param casing: The dictionary of tokens counter from `train()`.
        :type casing: default(Counter)
        """
        # Write a fresh sibling file and rename it over the target, rather than
        # truncating the target in place. Three things follow from that: a
        # symlink at `filename` is replaced instead of being followed to
        # whatever it points at (CWE-59), a crash mid-write cannot leave a
        # half-written model behind, and the check-then-write race in the CLI
        # ("if not os.path.isfile(modelfile): ... save_model(modelfile)")
        # can no longer clobber someone else's file. O_EXCL|O_NOFOLLOW on the
        # temporary name makes the create itself unspoofable; 0600 keeps the
        # corpus vocabulary out of a world-readable file (CWE-732).
        directory = os.path.dirname(os.path.abspath(filename))
        tmpname = os.path.join(
            directory, ".%s.%d.tmp" % (os.path.basename(filename), os.getpid())
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)  # not defined on Windows
        try:
            handle = os.open(tmpname, flags, 0o600)
        except FileExistsError:
            os.unlink(tmpname)
            handle = os.open(tmpname, flags, 0o600)
        try:
            with open(handle, "w", encoding=self.encoding) as fout:
                self._write_casing(casing, fout)
            os.replace(tmpname, filename)
        except BaseException:
            # Never leave the scratch file behind on an error path.
            try:
                os.unlink(tmpname)
            except OSError:
                pass
            raise

    def _write_casing(self, casing, fout):
        # The header marks a file whose every variant carries its true count,
        # i.e. one that is safe to merge. See MODEL_HEADER.
        print(MODEL_HEADER, file=fout)
        for token in casing:
            total_token_count = sum(casing[token].values())
            tokens_counts = []
            for i, (token, count) in enumerate(self._ranked_variants(casing[token])):
                if i == 0:
                    out_token = "{} ({}/{})".format(token, count, total_token_count)
                else:
                    out_token = "{} ({})".format(token, count, total_token_count)
                tokens_counts.append(out_token)
            print(" ".join(tokens_counts), end="\n", file=fout)

    def _load_model(self, filename):
        """
        Loads pre-trained truecasing file.

        Legacy files in the original Moses/Perl layout remain fully readable
        here: they simply lack the v2 header, and a non-best variant's ``(1)``
        is the Perl placeholder rather than a recoverable count. Such a file can
        still drive :meth:`truecase`, but :meth:`merge_models` rejects it with a
        :class:`LegacyModelError`.

        :returns: A dictionary of the best, known objects as values from `_casing_to_model()`
        :rtype: {'best': dict, 'known': Counter}
        """
        casing = self._read_casing(filename)
        # Returns the best and known object from `_casing_to_model()`
        return self._casing_to_model(casing)

    @staticmethod
    def _parse_count_field(field, filename, lineno):
        """Parse one ``(count)`` or ``(count/total)`` field into its integer count.

        The ``/total`` suffix only annotates the best variant on a line and is
        deliberately ignored: totals are recomputed from the per-variant counts
        on write, and trusting a stored total would let a hand-edited file carry
        a total that disagrees with the counts it lists.
        """
        try:
            return int(field.split("/")[0].strip("()"))
        except ValueError:
            raise ValueError(
                "malformed truecase model %r: line %d has a "
                "non-integer count %r" % (filename, lineno, field)
            ) from None

    def _read_casing(self, filename, require_counts=False):
        """Read a model file into a ``defaultdict(Counter)`` of variant counts.

        :param require_counts: When True, raise :class:`LegacyModelError` for a
            headerless file whose per-variant counts cannot be trusted for
            merging. Loading for inference passes False and reads anything.
        """
        casing = defaultdict(Counter)
        with open(filename, encoding=self.encoding) as fin:
            first_nonblank = True
            saw_header = False
            for lineno, line in enumerate(fin, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    # Only the header on the first non-blank line is recognised
                    # as a format marker; later comment lines are just skipped.
                    if first_nonblank and stripped == MODEL_HEADER:
                        saw_header = True
                    first_nonblank = False
                    continue
                first_nonblank = False
                fields = stripped.split()
                for token, count in grouper(fields, 2):
                    # `grouper` pads a short final pair with None, so a line
                    # with an odd number of fields used to surface as
                    # `AttributeError: 'NoneType' object has no attribute
                    # 'split'`, and a non-numeric count as a bare ValueError
                    # from int(). A downloaded/shared .truemodel is untrusted
                    # input, so report where it is malformed instead.
                    if count is None:
                        raise ValueError(
                            "malformed truecase model %r: line %d has an odd "
                            "number of fields (expected 'token (count/total)' "
                            "pairs)" % (filename, lineno)
                        )
                    count = self._parse_count_field(count, filename, lineno)
                    casing[token.lower()][token] += count
                if len(casing) > self.MAX_MODEL_ENTRIES:
                    # Loading costs roughly 35x the file size in RSS, so an
                    # oversized model is a memory-exhaustion vector (CWE-400).
                    # Raise MAX_MODEL_ENTRIES if you genuinely have a bigger
                    # vocabulary than this.
                    raise ValueError(
                        "truecase model %r exceeds MAX_MODEL_ENTRIES (%d) at "
                        "line %d" % (filename, self.MAX_MODEL_ENTRIES, lineno)
                    )
        if require_counts and not saw_header:
            # A headerless file is the Moses/Perl layout: every non-best variant
            # is stored as the placeholder `(1)`, so its real frequency cannot
            # be recovered and adding the counts would silently invent data.
            raise LegacyModelError(
                "cannot merge %r: it has no %r header, so its per-variant "
                "counts are not recoverable (the original Moses/Perl format "
                "stores every non-best variant as (1)). It can still be loaded "
                "for truecasing; re-train this shard with this sacremoses "
                "version to produce a mergeable model."
                % (filename, MODEL_HEADER)
            )
        return casing

    def merge_models(self, model_files, save_to=None, is_asr=None):
        """Merge truecaser model shards by adding their true variant counts.

        Each input is a model file produced by :meth:`save_model` (or the
        ``train-truecase`` CLI) -- typically one shard per batch of the corpus.
        The original corpus does not have to be reassembled: the per-variant
        counts already written into the shard files are summed instead.

        The counts of every surface form are added first, and only afterwards is
        the best (most frequent) form chosen via :meth:`_ranked_variants`. Because
        integer addition is associative and commutative and the tie-break is a
        fixed lexicographic rule, splitting into more/fewer batches or merging the
        shards in any order yields the identical merged model.

        Every input must carry the :data:`MODEL_HEADER`, i.e. its variant counts
        must be recoverable; a headerless legacy file raises
        :class:`LegacyModelError` instead of being merged on placeholder counts.

        :param model_files: Paths of the count-bearing model shards to merge.
        :type model_files: list(str)

        :param save_to: Optional path to write the merged count-bearing model.
        :type save_to: str or None

        :param is_asr: Override the ASR flag for building the in-memory model.
            Defaults to this instance's ``self.is_asr``.
        :type is_asr: bool or None

        :returns: The merged ``{'best', 'known', 'casing'}`` model and sets it on
            this instance as ``self.model``.
        :rtype: dict
        """
        if not model_files:
            raise ValueError(
                "merge_models() requires at least one model file; got an empty list"
            )
        merged = defaultdict(Counter)
        for model_file in model_files:
            shard = self._read_casing(model_file, require_counts=True)
            for token_lower, variant_counts in shard.items():
                # `Counter.__iadd__` drops non-positive results; every parsed
                # count is a non-negative integer, so a plain loop keeps the
                # addition explicit and the semantics obvious.
                for surface, count in variant_counts.items():
                    merged[token_lower][surface] += count

        previous_is_asr = self.is_asr
        if is_asr is not None:
            self.is_asr = is_asr
        try:
            self.model = self._casing_to_model(merged)
        finally:
            self.is_asr = previous_is_asr

        if save_to:
            self._save_model_from_casing(merged, save_to)
        return self.model


class MosesDetruecaser(object):
    def __init__(self):
        # Initialize the object.
        super(MosesDetruecaser, self).__init__()
        self.SENT_END = {".", ":", "?", "!"}
        self.DELAYED_SENT_START = {
            "(",
            "[",
            '"',
            "'",
            "&apos;",
            "&quot;",
            "&#91;",
            "&#93;",
        }

        # Some predefined tokens that will always be in lowercase.
        self.ALWAYS_LOWER = {
            "a",
            "after",
            "against",
            "al-.+",
            "and",
            "any",
            "as",
            "at",
            "be",
            "because",
            "between",
            "by",
            "during",
            "el-.+",
            "for",
            "from",
            "his",
            "in",
            "is",
            "its",
            "last",
            "not",
            "of",
            "off",
            "on",
            "than",
            "the",
            "their",
            "this",
            "to",
            "was",
            "were",
            "which",
            "will",
            "with",
        }

    def detruecase(self, text, is_headline=False, return_str=False):
        """
        Detruecase the translated files from a model that learnt from truecased
        tokens.

        :param text: A single string, i.e. sentence text.
        :type text: str
        """
        # `cased_tokens` keep tracks of detruecased tokens.
        cased_tokens = []
        sentence_start = True
        # Capitalize token if it's at the sentence start.
        for token in text.split():
            token = token[:1].upper() + token[1:] if sentence_start else token
            cased_tokens.append(token)
            if token in self.SENT_END:
                sentence_start = True
            elif not token in self.DELAYED_SENT_START:
                sentence_start = False
        # Check if it's a headline, if so then use title case.
        if is_headline:
            cased_tokens = [
                token if token in self.ALWAYS_LOWER else token[:1].upper() + token[1:]
                for token in cased_tokens
            ]

        return " ".join(cased_tokens) if return_str else cased_tokens


__all__ = ["MosesTruecaser", "MosesDetruecaser", "LegacyModelError", "MODEL_HEADER"]
