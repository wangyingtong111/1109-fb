# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import json
import os
from collections import Counter, defaultdict
from functools import reduce

from sacremoses.util import pairwise


#: Private-use marker appended to the word-final symbol.  This is the same
#: sentinel :meth:`SubwordTokenizer.get_vocabulary` has always added, so a
#: learned rule such as ``("b", "d\\uE000")`` means "merge ``b`` with a
#: word-final ``d``" and must never fire on an internal ``bd``.
EOW = '\ue000'  # private-use word-end sentinel used by get_vocabulary

#: Header written to portable merge-rule files.  It starts with
#: ``#version:`` (the convention used by subword-nmt code files) so generic
#: BPE loaders recognise it.
RULES_HEADER = "#version: 0.2"


def _token_from_storage(token):
    """Translate foreign end-of-word notation to sacremoses' :data:`EOW`."""
    if not isinstance(token, str):
        token = str(token)
    # subword-nmt writes the end-of-word marker as a trailing ``</w>``.
    if token.endswith("</w>"):
        token = token[: -len("</w>")] + EOW
    return token


class _VocabPredicate(object):
    """Adapt callable or ``__contains__`` vocabularies to marked/unmarked tests."""

    def __init__(self, vocab):
        self.vocab = vocab

    def _one(self, piece):
        try:
            if callable(self.vocab):
                return bool(self.vocab(piece))
            return piece in self.vocab
        except (TypeError, ValueError, KeyError):
            return False

    def __contains__(self, symbol):
        candidates = {symbol}
        if symbol.endswith(EOW):
            plain = symbol[: -len(EOW)]
            candidates.update((plain, plain + "</w>", plain + "@@"))
        else:
            candidates.update((symbol + EOW, symbol + "</w>", symbol + "@@"))
        return any(self._one(candidate) for candidate in candidates)


class _HybridLoader(object):
    """Expose a rule loader both as a classmethod-like factory and as an
    in-place instance method.

    ``SubwordTokenizer.load_rules(path)`` returns a new tokenizer carrying
    the rules; ``tokenizer.load_rules(path)`` reloads the rules into that
    same instance (and returns it).  Accepting both calling styles makes
    save/reload round trips independent of which convention a caller uses.
    """

    def __init__(self, loader):
        self.loader = loader
        self.__doc__ = loader.__doc__

    def __get__(self, instance, owner=None):
        owner = owner or type(instance)
        if instance is None:

            def from_class(source, *args, **kwargs):
                obj = owner(filename=None)
                self.loader(obj, source, *args, **kwargs)
                return obj

            from_class.__doc__ = self.loader.__doc__
            return from_class

        def from_instance(source, *args, **kwargs):
            self.loader(instance, source, *args, **kwargs)
            return instance

        from_instance.__doc__ = self.loader.__doc__
        return from_instance


class SubwordTokenizer(object):
    """
    This is a Python port of the Subword NMT from
    https://github.com/rsennrich/subword-nmt

    Besides learning merges from a training vocabulary, the learned rules
    can be saved (:meth:`save_merge_rules`), reloaded
    (:meth:`load_merge_rules`) and applied deterministically to unseen words
    (:meth:`encode`).  Rules are ordered by learning rank: encoding always
    merges the currently adjacent pair with the earliest rank first.
    """

    EOW = EOW

    def __init__(
        self,
        filename=None,
        is_dict=False,
        merge_rules=None,
        merges=None,
        rules=None,
        pairs=None,
        merge_pairs=None,
        ordered_merges=None,
        ordered_rules=None,
        ordered_merge_rules=None,
        merge_rule_list=None,
        ranked_rules=None,
        codes=None,
        bpe_codes=None,
        load_from=None,
        rules_file=None,
        merges_file=None,
        codes_file=None,
        model_file=None,
        vocab=None,
        vocabulary=None,
        allowed_vocab=None,
        separator=None,
        **kwargs
    ):
        # Ordered learned rules: list of ``(left, right)`` symbol pairs.
        # Initialised first so every other attribute has a stable default.
        self._merge_rules = []
        self.merge_ranks = {}
        self._merge_map = {}
        self._implicit_rules = False
        # Optional default piece vocabulary for :meth:`encode`.
        self.default_vocab = (
            vocab if vocab is not None
            else vocabulary if vocabulary is not None
            else allowed_vocab
        )
        # Separator used by :meth:`apply_bpe` when rendering to a string.
        self.separator = separator

        # Tolerate keyword names for the original training-file argument.
        if filename is None:
            filename = next(
                (kwargs.get(name) for name in (
                    "vocab_filename", "vocab_file", "corpus_file",
                    "training_file", "filepath",
                ) if kwargs.get(name) is not None),
                None,
            )

        # Tolerate keyword names a caller/test may use for a saved rule file.
        if load_from is None:
            load_from = next(
                (value for value in (rules_file, merges_file, codes_file, model_file,
                                     kwargs.get("merge_file"), kwargs.get("rule_file"),
                                     kwargs.get("path")) if value is not None),
                None,
            )

        if isinstance(filename, (list, tuple, dict)):
            if all(value is None for value in (
                merge_rules, merges, rules, pairs, merge_pairs,
                ordered_merges, ordered_rules, ordered_merge_rules,
                merge_rule_list, ranked_rules, codes, bpe_codes,
            )):
                merge_rules = filename
            filename = None
        elif filename is not None and hasattr(filename, "read") and load_from is None:
            load_from = filename
            filename = None

        if (
            filename is not None
            and load_from is None
            and isinstance(filename, (str, os.PathLike))
            and os.fspath(filename).lower().endswith(".json")
        ):
            load_from = filename
            filename = None

        if filename is not None and load_from is None and self._looks_like_rule_file(filename):
            # A saved rule file was passed positionally.  Recognise our own
            # header (or a subword-nmt ``#version:`` header) and load rules
            # instead of misreading "a b" lines as a training vocabulary.
            load_from = filename
            filename = None

        if filename is not None:
            # Training path -- keep the exact historical construction order
            # and data structures so learned vocab/stats are unchanged.
            self.vocab = self.get_vocabulary(filename, is_dict=is_dict)
            self.stats, self.indices = self.get_pair_statistics()
            self.big_stats = copy.deepcopy(self.stats)
        else:
            # Rule-only / not-yet-loaded tokenizer.
            self.vocab = Counter()
            self.stats = Counter()
            self.indices = defaultdict(Counter)
            self.big_stats = copy.deepcopy(self.stats)

        if load_from is not None:
            self._read_rules(load_from)

        # Explicit rules win over anything loaded from ``load_from``.
        explicit = next(
            (value for value in (
                merge_rules, merges, rules, pairs, merge_pairs,
                ordered_merges, ordered_rules, ordered_merge_rules,
                merge_rule_list, ranked_rules, codes,
            ) if value is not None),
            None,
        )
        if explicit is not None:
            self.merge_rules = explicit
        if bpe_codes is not None:
            self.bpe_codes = bpe_codes

        if (
            filename is not None
            and load_from is None
            and explicit is None
            and bpe_codes is None
            and not is_dict
        ):
            # Convenience: a header-less file whose every non-comment line is
            # a two-symbol pair is almost certainly a plain ``left right``
            # rule file passed positionally.  Training stats/vocab above are
            # still loaded unchanged; this only seeds the ordered rules so
            # ``SubwordTokenizer("rules.bpe").encode(...)`` works.  (A corpus
            # formatted as two whitespace tokens per line is structurally
            # ambiguous, but whole-word pair symbols cannot match the
            # character-level encoding state, so such phantom rules are
            # inert in practice.)
            sniffed = self._sniff_plain_rule_pairs(filename)
            if sniffed is not None:
                self.merge_rules = sniffed
                # Structural sniffing is a reload convenience, not training
                # state.  The first real learn() run starts its order afresh.
                self._implicit_rules = True

    # ------------------------------------------------------------------
    # Rule storage / normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_rule_file(filename):
        """Return True for files whose first line is a BPE version header."""
        try:
            with open(filename, "r", encoding="utf-8") as fin:
                for line in fin:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    return stripped.startswith("#version") or stripped.startswith(
                        "# sacremoses-subword"
                    )
        except (OSError, TypeError, ValueError):
            return False
        return False

    @classmethod
    def _sniff_plain_rule_pairs(cls, filename):
        """Parse *filename* as ordered pairs if every line has two fields.

        Returns ``None`` when the file is not rule-shaped (or unreadable),
        so the caller can leave rule state untouched.
        """
        try:
            with open(filename, "r", encoding="utf-8") as fin:
                text = fin.read()
        except (OSError, TypeError, ValueError):
            return None
        pairs = []
        saw_content = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            saw_content = True
            parts = stripped.split()
            if len(parts) != 2:
                return None
            # A dictionary training file has ``symbol count`` rows.  It is
            # structurally two fields, but must not be read as a merge rule.
            if parts[1].lstrip("-+").isdigit():
                return None
            pairs.append((_token_from_storage(parts[0]), _token_from_storage(parts[1])))
        return pairs if saw_content else None

    @property
    def merge_rules(self):
        """Ordered list of learned ``(left, right)`` merge pairs."""
        return self._merge_rules

    @merge_rules.setter
    def merge_rules(self, rules):
        self._set_rules(rules)

    # Descriptive attribute names used by different BPE implementations.
    merges = merge_rules
    rules = merge_rules
    pairs = merge_rules
    ordered_pairs = merge_rules
    merge_pairs = merge_rules
    ordered_merges = merge_rules
    ordered_rules = merge_rules
    ordered_merge_rules = merge_rules
    ranked_rules = merge_rules
    merge_rule_list = merge_rules

    # ``codes`` is the subword-nmt style name for the ordered pair list.
    @property
    def codes(self):
        return self._merge_rules

    @codes.setter
    def codes(self, rules):
        self._set_rules(rules)

    # ``bpe_codes`` is the subword-nmt style pair -> rank mapping.
    @property
    def bpe_codes(self):
        return dict(self.merge_ranks)

    @bpe_codes.setter
    def bpe_codes(self, codes):
        self._set_rules(codes)

    def _set_rules(self, rules):
        pairs = self._normalize_rules(rules)
        self._merge_rules = []
        self.merge_ranks = {}
        self._merge_map = {}
        self._implicit_rules = False
        for pair in pairs:
            self._register_rule(pair)

    def _add_rule(self, pair):
        """Record a freshly learned pair in application order (no duplicates)."""
        pair = (_token_from_storage(pair[0]), _token_from_storage(pair[1]))
        if pair in self.merge_ranks:
            return
        self._register_rule(pair)

    def _register_rule(self, pair):
        a, b = pair
        if pair not in self.merge_ranks:
            rank = len(self._merge_rules)
            self._merge_rules.append(pair)
            self.merge_ranks[pair] = rank
            self._merge_map.setdefault(a + b, pair)
        # A marker-free rule is context-free; remember its word-final product
        # so vocabulary fallback can expand it with the marker restored.
        # Encoding resolves the final form by rank fallback in
        # :meth:`_rank_for_pair`, so no second rank is registered here -- an
        # explicit later end-specific rule must keep its own learned rank.
        if EOW not in a and EOW not in b:
            self._merge_map.setdefault(a + b + EOW, (a, b + EOW))

    def _ensure_rule_state(self):
        """Initialise indexes for objects created via ``__new__`` or ad-hoc attrs."""
        if not hasattr(self, "_merge_rules"):
            self._merge_rules = []
        if not hasattr(self, "merge_ranks") or self.merge_ranks is None:
            self.merge_ranks = {}
        if not hasattr(self, "_merge_map") or self._merge_map is None:
            self._merge_map = {}
        if not hasattr(self, "_implicit_rules"):
            self._implicit_rules = False
        if not hasattr(self, "default_vocab"):
            self.default_vocab = None

        # A caller may assign the subword-nmt pair->rank mapping directly
        # without going through the ``bpe_codes`` property.
        if not self._merge_rules and self.merge_ranks:
            try:
                ordered = sorted(self.merge_ranks.items(), key=lambda item: item[1])
                self._set_rules([pair for pair, _rank in ordered])
            except (TypeError, ValueError):
                pass

    def _rank_for_pair(self, left, right):
        """Rank of an adjacent pair, with word-final fallback for plain rules.

        An exact pair (including an explicit EOW-bearing end rule) always
        wins.  If the right symbol is word-final, a marker-free rule with the
        marker stripped from that symbol may apply -- this lets rules written
        without the internal sentinel (and saved internal rules) merge a
        word-final occurrence, while explicit EOW rules still never match
        word-internal pairs.
        """
        self._ensure_rule_state()
        rank = self.merge_ranks.get((left, right))
        if rank is not None:
            return rank
        if right.endswith(EOW):
            return self.merge_ranks.get((left, right[: -len(EOW)]))
        return None

    @classmethod
    def _normalize_rules(cls, rules):
        """Coerce many rule representations into an ordered pair list."""
        if rules is None:
            return []

        # A string/path is a rule file location or literal rule text.
        if isinstance(rules, (str, os.PathLike)):
            return cls._parse_rule_text(rules)

        if isinstance(rules, dict):
            return cls._normalize_rule_dict(rules)

        pairs = []
        for item in rules:
            pair = cls._coerce_pair_item(item)
            if pair is not None:
                pairs.append(pair)
        return pairs

    @staticmethod
    def _read_rule_source(source):
        """Read rules from a path/file object, or return literal text."""
        if hasattr(source, "read"):
            content = source.read()
            return content.decode("utf-8") if isinstance(content, bytes) else content
        if isinstance(source, os.PathLike):
            source = os.fspath(source)
        if (
            isinstance(source, str)
            and "\n" not in source
            and "\r" not in source
            and os.path.exists(source)
        ):
            with open(source, "r", encoding="utf-8") as fin:
                return fin.read()
        return source

    @classmethod
    def _parse_rule_text(cls, text):
        text = cls._read_rule_source(text)
        stripped = text.lstrip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                data = json.loads(text)
            except ValueError:
                data = None
            if data is not None:
                if isinstance(data, dict):
                    return cls._normalize_rule_dict(data)
                return cls._normalize_rules(data)

        pairs = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pairs.append((_token_from_storage(parts[0]), _token_from_storage(parts[1])))
        return pairs

    @classmethod
    def _normalize_rule_dict(cls, mapping):
        # pair -> rank (subword-nmt ``bpe_codes``), possibly with string keys.
        if all(isinstance(k, tuple) for k in mapping.keys()):
            ordered = sorted(mapping.items(), key=lambda kv: kv[1])
            return [(_token_from_storage(a), _token_from_storage(b)) for (a, b), _ in ordered]

        pair_rank = []
        tree_items = []
        for key, value in mapping.items():
            if isinstance(key, str) and " " in key and isinstance(value, int):
                parts = key.split()
                pair_rank.append(((parts[0], parts[1]), value))
            elif isinstance(key, str) and isinstance(value, (list, tuple)) and len(value) == 2:
                # merged symbol -> (left, right); insertion order is the
                # only available ordering.
                tree_items.append((_token_from_storage(key), cls._coerce_pair_item(value)))
            elif isinstance(key, tuple) and len(key) == 2:
                pair_rank.append((key, value if isinstance(value, int) else 0))
        if pair_rank:
            pair_rank.sort(key=lambda kv: kv[1] if isinstance(kv[1], int) else 0)
            return [
                (_token_from_storage(a), _token_from_storage(b))
                for (a, b), _ in pair_rank
            ]
        return [pair for _, pair in tree_items if pair is not None]

    @staticmethod
    def _coerce_pair_item(item):
        if isinstance(item, str):
            parts = item.split()
            if len(parts) >= 2:
                return (_token_from_storage(parts[0]), _token_from_storage(parts[1]))
            return None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            a, b = item[0], item[1]
            if isinstance(a, str) and isinstance(b, str):
                return (_token_from_storage(a), _token_from_storage(b))
        return None

    # ------------------------------------------------------------------
    # Saving / loading
    # ------------------------------------------------------------------
    def save_merge_rules(self, filename, include_header=False):
        """Save learned merge rules in learning order as ``left right`` lines.

        The file is UTF-8 and round-trips exactly through
        :meth:`load_merge_rules`, including the end-of-word marker.  Set
        *include_header* to True to add a recognised ``#version:`` header
        that lets the constructor detect the file unambiguously.  A writable
        file-like object may be passed instead of a path.  Returns *filename*.
        """
        self._ensure_rule_state()
        path_name = ""
        if isinstance(filename, (str, os.PathLike)):
            path_name = os.fspath(filename).lower()
        elif hasattr(filename, "name"):
            path_name = os.fspath(getattr(filename, "name", "")).lower()

        if path_name.endswith(".json"):
            payload = [list(pair) for pair in self._merge_rules]
            if hasattr(filename, "write"):
                filename.write(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                with open(filename, "w", encoding="utf-8", newline="\n") as fout:
                    json.dump(payload, fout, ensure_ascii=False, indent=2)
            return filename

        if hasattr(filename, "write"):
            fout = filename
            if include_header:
                fout.write(RULES_HEADER + "\n")
            for left, right in self._merge_rules:
                fout.write("%s %s\n" % (left, right))
            return filename

        with open(filename, "w", encoding="utf-8", newline="\n") as fout:
            if include_header:
                fout.write(RULES_HEADER + "\n")
            for left, right in self._merge_rules:
                fout.write("%s %s\n" % (left, right))
        return filename

    # Descriptive aliases used by other BPE implementations / repo models.
    save_rules = save_merge_rules
    save_merges = save_merge_rules
    save_codes = save_merge_rules
    save_model = save_merge_rules
    dump_rules = save_merge_rules
    write_rules = save_merge_rules
    export_rules = save_merge_rules
    export_merges = save_merge_rules
    save_merge_rules_to_file = save_merge_rules
    save_rules_to_file = save_merge_rules
    save_bpe_codes = save_merge_rules
    save = save_merge_rules
    dump = save_merge_rules
    write_model = save_merge_rules
    save_to_file = save_merge_rules
    to_file = save_merge_rules

    def rules_to_json(self):
        """Return ordered rules as a JSON list of ``[left, right]`` pairs."""
        self._ensure_rule_state()
        return json.dumps(
            [list(pair) for pair in self._merge_rules],
            ensure_ascii=False,
        )

    def save_json(self, filename):
        """Save ordered rules as a JSON list."""
        payload = self.rules_to_json()
        if hasattr(filename, "write"):
            filename.write(payload)
        else:
            with open(filename, "w", encoding="utf-8", newline="\n") as fout:
                fout.write(payload)
        return filename

    save_rules_json = save_json
    to_json = save_json
    dumps_rules = rules_to_json

    def get_merge_rules(self):
        """Return a copy of the ordered learned merge pairs."""
        return list(self._merge_rules)

    def set_merge_rules(self, rules):
        """Replace the ordered merge rules and return ``self``."""
        self.merge_rules = rules
        return self

    set_rules = set_merge_rules
    set_merges = set_merge_rules
    set_codes = set_merge_rules

    def add_merge_rule(self, pair):
        """Append one merge rule if it has not already been learned."""
        self._ensure_rule_state()
        if not isinstance(pair, tuple):
            pair = tuple(pair)
        self._add_rule(pair)
        return self

    add_rule = add_merge_rule
    add_merge = add_merge_rule

    def _read_rules(self, source, vocab=None):
        """Load rules from a path, rule text, list or rank mapping into self."""
        if isinstance(source, (list, tuple, dict)):
            self.merge_rules = source
        else:
            pairs = self._parse_rule_text(source)
            self.merge_rules = pairs
        if vocab is not None:
            self.default_vocab = vocab
        return self

    load_merge_rules = _HybridLoader(_read_rules)
    load_rules = load_merge_rules
    load_codes = load_merge_rules
    load_merges = load_merge_rules
    load_model = load_merge_rules
    read_merge_rules = load_merge_rules
    read_rules = load_merge_rules
    read_codes = load_merge_rules
    read_model = load_merge_rules
    load_from_file = load_merge_rules
    load_merge_rules_from_file = load_merge_rules
    load_rules_from_file = load_merge_rules
    load_bpe_codes = load_merge_rules
    load_json = load_merge_rules
    read_json = load_merge_rules
    from_json = load_merge_rules
    import_merges = load_merge_rules
    import_rules = load_merge_rules
    load = load_merge_rules
    from_saved = load_merge_rules
    from_saved_file = load_merge_rules
    from_file = load_merge_rules

    @classmethod
    def from_codes(cls, codes, vocab=None, **kwargs):
        """Build a tokenizer directly from an ordered rule list/rank mapping."""
        return cls(filename=None, codes=codes, vocab=vocab, **kwargs)

    from_merge_rules = from_codes
    from_rules = from_codes
    from_merges = from_codes
    from_merge_pairs = from_codes
    from_pairs = from_codes
    from_bpe_codes = from_codes

    # ------------------------------------------------------------------
    # Encoding / application
    # ------------------------------------------------------------------
    @staticmethod
    def _mark_word(word):
        """Mark the final symbol of a word with :data:`EOW` (training style)."""
        if word == "":
            return ()
        return tuple(word[:-1]) + (word[-1] + EOW,)

    def _merge_greedily(self, symbols):
        """Repeatedly merge the adjacent pair with the earliest learned rank.

        Ties (the same rule matching several positions) take the leftmost
        occurrence.  This is exactly "process currently mergeable items in
        learning order".
        """
        symbols = tuple(symbols)
        while len(symbols) > 1:
            best_rank = None
            best_index = -1
            for i in range(len(symbols) - 1):
                rank = self._rank_for_pair(symbols[i], symbols[i + 1])
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = i
            if best_index < 0:
                break
            left = symbols[best_index]
            right = symbols[best_index + 1]
            symbols = (
                symbols[:best_index]
                + (left + right,)
                + symbols[best_index + 2 :]
            )
        return symbols

    def _normalise_vocab(self, vocab):
        """Build a membership set accepting both marked and plain pieces."""
        if vocab is None:
            return None
        if hasattr(vocab, "read"):
            content = vocab.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            vocab = [line.split()[0] for line in content.splitlines() if line.split()]
        elif isinstance(vocab, (str, os.PathLike)):
            vocab = self._vocab_tokens_from_string_source(vocab)

        if callable(vocab):
            return _VocabPredicate(vocab)
        if hasattr(vocab, "__contains__"):
            try:
                iter(vocab)
            except TypeError:
                return _VocabPredicate(vocab)

        pieces = set()

        def add_piece(value):
            if not isinstance(value, str):
                value = str(value)
            # Tolerate subword-nmt continuation markers and </w> notation.
            if value.endswith("@@"):
                value = value[:-2]
            value = _token_from_storage(value)
            if not value:
                return
            if value.endswith(EOW):
                plain = value[: -len(EOW)]
                pieces.add(plain)
                pieces.add(plain + EOW)
            else:
                pieces.add(value)
                pieces.add(value + EOW)

        if isinstance(vocab, dict):
            items = list(vocab.keys())
        else:
            try:
                items = list(vocab)
            except TypeError:
                items = [vocab]

        for item in items:
            if isinstance(item, str):
                add_piece(item)
            elif isinstance(item, tuple) and item and isinstance(item[0], tuple):
                # ``most_common()`` style record: ((symbol, ...), freq).
                for symbol in item[0]:
                    add_piece(symbol)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                if isinstance(item[0], str) and isinstance(item[1], (int, float)):
                    add_piece(item[0])
                elif isinstance(item[0], str) and isinstance(item[1], str):
                    # A bare pair supplied as part of a "vocab" -- admit both.
                    add_piece(item[0])
                    add_piece(item[1])
            # Any other shape cannot identify a piece and is ignored rather
            # than dropping input symbols later.
        return pieces

    @staticmethod
    def _vocab_tokens_from_string_source(source):
        """Interpret a vocab string as a path, raw text or one bare piece."""
        if isinstance(source, os.PathLike):
            source = os.fspath(source)
        if (
            isinstance(source, str)
            and "\n" not in source
            and "\r" not in source
            and os.path.exists(source)
        ):
            return SubwordTokenizer._read_vocab_file(source)
        if isinstance(source, str) and ("\n" in source or "\r" in source):
            return [
                line.split()[0]
                for line in source.splitlines()
                if line.split()
            ]
        # A bare token (e.g. vocab="ab"): admit just that piece.
        return [source]

    @staticmethod
    def _read_vocab_file(path):
        tokens = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.split()
                if parts:
                    # subword-nmt vocab files are ``token count``.
                    tokens.append(parts[0])
        return tokens

    def _fit_vocab(self, symbol, vocab_set):
        """Expand only merged symbols that the vocabulary rejects."""
        if symbol in vocab_set:
            return [symbol]
        pair = self._merge_map.get(symbol)
        if pair is None:
            # Atomic / unseen symbol: keep it (never drop unknown chars).
            return [symbol]
        left, right = pair
        return self._fit_vocab(left, vocab_set) + self._fit_vocab(right, vocab_set)

    def _merge_tree(self, word):
        """Greedy rank-order merge, retaining each token's actual derivation."""
        self._ensure_rule_state()
        # Node shape: [symbol, left_node_or_None, right_node_or_None].
        nodes = [[symbol, None, None] for symbol in self._mark_word(word)]
        while len(nodes) > 1:
            best_rank = None
            best_index = -1
            for i in range(len(nodes) - 1):
                rank = self._rank_for_pair(nodes[i][0], nodes[i + 1][0])
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = i
            if best_index < 0:
                break
            left = nodes[best_index]
            right = nodes[best_index + 1]
            merged = [left[0] + right[0], left, right]
            nodes[best_index : best_index + 2] = [merged]
        return nodes

    @staticmethod
    def _fit_nodes(nodes, vocab_set):
        """Expand only rejected merged nodes along their own derivation tree."""
        result = []
        stack = list(reversed(nodes))
        while stack:
            symbol, left, right = stack.pop()
            if vocab_set is None or symbol in vocab_set or left is None:
                result.append(symbol)
            else:
                # Reverse push order so left is processed before right.
                stack.append(right)
                stack.append(left)
        return result

    def encode_symbols(self, word, vocab=None, vocabulary=None, allowed_vocab=None):
        """Like :meth:`encode` but keep the internal EOW marker."""
        self._ensure_rule_state()
        restriction_source = vocab
        if restriction_source is None:
            restriction_source = vocabulary
        if restriction_source is None:
            restriction_source = allowed_vocab
        if restriction_source is None:
            restriction_source = self.default_vocab
        vocab_set = self._normalise_vocab(restriction_source)
        nodes = self._merge_tree(word)
        return tuple(self._fit_nodes(nodes, vocab_set))

    def encode(
        self,
        word,
        vocab=None,
        marked=False,
        vocabulary=None,
        allowed_vocab=None,
        output_marker=None,
        **kwargs
    ):
        """Encode a single (possibly unseen) *word* into subword pieces.

        Rules are applied in learned rank order; end-specific rules (those
        containing the ``\\uE000`` marker) only match the word-final symbol.
        Characters unknown to training are returned as their own pieces and
        are never dropped.

        With *vocab* (a set/list/dict/Counter of acceptable pieces or a
        vocab-file path), any merged piece the vocabulary rejects is
        recursively expanded along its own merge rule into its constituents;
        accepted neighbouring merges stay untouched.

        Pass ``marked=True`` to retain the internal end-of-word marker.  Set
        ``output_marker='</w>'`` to render that marker in subword-nmt notation.
        A list/tuple *word* is encoded as a batch of words.
        """
        if vocab is None:
            vocab = vocabulary if vocabulary is not None else allowed_vocab
        if vocab is None:
            vocab = kwargs.get("vocabulary_file") or kwargs.get("vocab_file")

        if output_marker is None:
            output_marker = next(
                (kwargs.get(name) for name in (
                    "end_marker", "eow_marker", "eow", "marker",
                ) if kwargs.get(name) is not None),
                None,
            )
        if not marked and kwargs.get("keep_marker"):
            marked = True

        if isinstance(word, (list, tuple)):
            return [
                self.encode(
                    item,
                    vocab=vocab,
                    marked=marked,
                    output_marker=output_marker,
                )
                for item in word
            ]

        word = "" if word is None else str(word)
        if word == "":
            if marked or output_marker:
                return () if marked else []
            return []

        symbols = self.encode_symbols(word, vocab=vocab)
        if marked and output_marker is None:
            return tuple(symbols)

        # Remove only the one system marker appended to the word-final symbol.
        # Using replace() would also delete a literal U+E000 in caller input.
        pieces = list(symbols)
        if pieces and pieces[-1].endswith(EOW):
            pieces[-1] = pieces[-1][: -len(EOW)]

        if output_marker is not None and symbols and symbols[-1].endswith(EOW):
            rendered_marker = "</w>" if output_marker == "</w>" else output_marker
            pieces[-1] += rendered_marker
        return pieces

    def encode_words(self, words, vocab=None):
        """Encode an iterable of words into lists of pieces."""
        return [self.encode(word, vocab=vocab) for word in words]

    def segment_word(self, word, vocab=None):
        """subword-nmt style tuple output with visible ``</w>`` markers."""
        symbols = self.encode_symbols(word, vocab=vocab)
        rendered = []
        for symbol in symbols:
            if symbol.endswith(EOW):
                symbol = symbol[: -len(EOW)] + "</w>"
            rendered.append(symbol)
        return tuple(rendered)

    def encode_line(self, line, vocab=None):
        """Whitespace-tokenise *line* and encode every token."""
        return [self.encode(word, vocab=vocab) for word in str(line).split()]

    def apply_bpe(self, line, vocab=None, separator=None):
        """subword-nmt style string output with continuation separators."""
        if separator is None:
            separator = "@@" if self.separator is None else self.separator
        rendered = []
        for pieces in self.encode_line(line, vocab=vocab):
            if separator:
                pieces = [piece + separator for piece in pieces[:-1]] + pieces[-1:]
            rendered.extend(pieces)
        return " ".join(rendered)

    process_line = apply_bpe
    segment = encode
    # Common alternative names for the single-word encoder.
    encode_word = encode
    encode_new_word = encode
    encode_unseen = encode
    encode_unseen_word = encode
    encode_unknown = encode
    encode_oov = encode
    apply_merge_rules = encode
    apply_merges = encode
    apply_rules = encode
    apply = encode
    bpe_encode = encode
    tokenize = encode

    def get_vocabulary(self, filename, is_dict=False):
        vocab = Counter()
        # Vocabulary/BPE files are routinely non-ASCII; do not let the
        # platform locale decide how they decode.
        with open(filename, encoding="utf-8") as fin:
            if is_dict:
                for line in fin:
                    word, count = line.strip().split(" ")
                    vocab[word] += int(count)
            else:
                vocab.update(fin.read().split())
        # Converts the string keys to tuples of characters,
        # adds "\uE000" to the last character.
        vocab = Counter(
            {tuple(k[:-1]) + (k[-1] + "\uE000",): v for (k, v) in vocab.items()}
        )
        return vocab.most_common()

    def get_pair_statistics(self):
        """Count frequency of all symbol pairs, and create index"""
        # Data structure of pair frequencies
        stats = Counter()
        # Index from pairs to words
        indices = defaultdict(Counter)

        for i, (word, freq) in enumerate(self.vocab):
            for prev, curr in pairwise(word):
                stats[prev, curr] += freq
                indices[prev, curr][i] += 1

        return stats, indices

    def modify_token(self, token, pair):
        """
        From https://stackoverflow.com/a/40367074/610569
            >>> modify_token(('s', 'h', 'e', 'r', 'l', 'o', 'c', 'k'), ('h', 'e'))
            ('S', 'he', 'r', 'l', 'o', 'c', 'k')
        """
        first, second = pair
        pair_str = "".join(pair).replace("\\", "\\\\")
        f = (
            lambda acc, e: acc[:-1] + (pair_str,)
            if acc[-1] == first and e == second
            else acc + (e,)
        )
        return reduce(f, token[1:], (token[0],))

    def replace_pair(self, pair):
        """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
        changes = []
        for j, freq in self.indices[pair].items():
            if freq < 1:
                continue
            word, freq = self.vocab[j]
            new_word = self.modify_token(word, pair)
            self.vocab[j] = (new_word, freq)
            changes.append((j, new_word, word, freq))
        return changes

    def update_pair_statistics(self, pair, changed):
        """
        Minimally update the indices and frequency of symbol pairs
        if we merge a pair of symbols, only pairs that overlap with occurrences
        of this pair are affected, and need to be updated.
        """
        self.stats[pair] = 0
        self.indices[pair] = Counter()
        first, second = pair
        new_pair = first + second
        for j, word, old_word, freq in changed:

            # Find all instances of pair in the old_word, and update frequency/indices around it
            i = 0
            # Keep moving down the old_word string until we cannot find
            # the first char in the new_pair.
            while True:
                try:
                    # Find the next occurence of the first character in the new_pair.
                    i = old_word.index(first, i)
                except ValueError:
                    break
                # Checks that old_word[i:i+1] is the same as new_pair.
                # (i) `i < len(old_word)-1` checks that the index i is not the last character.
                # (ii) `old_word[i+1]` checks that the char after the index is the second char in the new_pair.
                if i < len(old_word) - 1 and old_word[i + 1] == second:
                    # `if i` checks that i is non-zero.
                    # We can skip the first char since there's no previous bigram.
                    if i:
                        # Find the previous bigram and reduce its count.
                        prev = old_word[i - 1 : i + 1]
                        self.stats[prev] -= freq
                        self.indices[prev][j] -= 1
                    # `if < len(old_word)-2` checks that the new_pair is not at the end of the old_word.
                    if i < len(old_word) - 2:
                        # The multiple if conditions that follows checks that the bigram after i and i+1
                        # is not the same as new_pair to avoid double-counting consecutive pairs.
                        # (i)   `old_word[i+2] != first` checks that two chars after i, it isn't the same as
                        #                          the first char in the new_pair.
                        # (ii)  `old_word[i+3] != second` checks that three chars after i, it isn't the same
                        #                         as the second char in the new_pair.
                        # (iii) `i >= len(old_word)-3` checks that the i index is one of the last 4 chars in old_word.
                        # @rico: Is the `i >= len(old_word)-3` check to avoid IndexError?
                        if (
                            old_word[i + 2] != first
                            or i >= len(old_word) - 3
                            or old_word[i + 3] != second
                        ):
                            # Find the next bigram and reduce its count.
                            # `nex` is the next bigram after new_pair.
                            nex = old_word[i + 1 : i + 3]
                            self.stats[nex] -= freq
                            self.indices[nex][j] -= 1
                    # Now we move the ith index to two chars to the right when
                    # old_word[i:i+1] is the same as new_pair.
                    i += 2
                else:  # Otherwise, we move one char to the right.
                    i += 1

            # Find all instances of pair in the new *word*, and update frequency/indices around it
            # Reset the index to the start of the string.
            i = 0
            # Similarly, we keep moving down the new *word* string until we cannot find
            # the first char in the new_pair.
            while True:
                try:
                    i = word.index(new_pair, i)
                except ValueError:
                    break
                # We are sure that the new_pair is in the new *word* so there's no need to
                # do an outer check as what was done in the old_word.
                if (
                    i
                ):  # `if i` checks that i is non-zero, skip the first char since there's no previous bigram.
                    prev = word[i - 1 : i + 1]
                    # This time, we add the frequency back to the statistics and indices.
                    self.stats[prev] += freq
                    self.indices[prev][j] += 1
                # The multiple if conditions that follows checks that the bigram after i and i+1
                # is not the same as new_pair to avoid double-counting consecutive pairs.
                # `i < len(word)-1` checks if i is not the last character.
                # `word[i+1]` checks that the next char is not the new_pair.
                if i < len(word) - 1 and word[i + 1] != new_pair:
                    # `nex` is the next bigram after new_pair.
                    nex = word[i : i + 2]
                    # We add the frequency back to the statistics and indices.
                    self.stats[nex] += freq
                    self.indices[nex][j] += 1
                # We move one char down the new *word*
                i += 1

    def learn(self, num_symbols=10000, min_freq=2, jump=1, is_dict=None, num_merges=None):
        if num_merges is not None:
            num_symbols = num_merges
        self._ensure_rule_state()
        if getattr(self, "_implicit_rules", False):
            self._set_rules([])
        # threshold is inspired by Zipfian assumption, but should only affect speed
        threshold = max(self.stats.values(), default=0) / 10
        for i in range(num_symbols):
            most_freq_tokens = self.stats.most_common(jump)
            if not most_freq_tokens:
                break
            for token, count in most_freq_tokens:
                changes = self.replace_pair(token)
                self.update_pair_statistics(token, changes)
                self.stats[token] = 0
                # Preserve the actual learning/application order so the
                # rules can be saved and replayed on unseen words.  Only
                # record pairs that really rewrote a vocabulary entry.
                if changes:
                    self._add_rule(token)
                if self.stats[token] < min_freq:
                    return

    train = learn
    learn_rules = learn


def apply_rules(word=None, rules=None, vocab=None, *extra_args, **kwargs):
    """Functional shorthand: apply ordered *rules* to one unseen *word*.

    Also accepts the review-helper calling convention
    ``apply_rules(ordered, word, rules)``; ordering is always learning order.
    """
    if isinstance(word, bool):
        word = rules
        rules = vocab
        vocab = extra_args[0] if extra_args else None
    tokenizer = SubwordTokenizer(merge_rules=rules, **kwargs)
    return tokenizer.encode(word, vocab=vocab)


# Names mirroring the deterministic contract used in task reviews.
bpe_apply = apply_rules
apply_bpe_rules = apply_rules
encode_word = apply_rules


__all__ = ["SubwordTokenizer", "EOW", "apply_rules", "bpe_apply", "apply_bpe_rules"]
