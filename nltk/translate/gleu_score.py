# Natural Language Toolkit: GLEU Score
#
# Copyright (C) 2001-2026 NLTK Project
# Authors:
# Contributors: Mike Schuster, Michael Wayne Goodman, Liling Tan
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""GLEU score implementation."""

from collections import Counter

from nltk.util import everygrams, ngrams


def sentence_gleu(references, hypothesis, min_len=1, max_len=4):
    """
    Calculates the sentence level GLEU (Google-BLEU) score described in

        Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi,
        Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey,
        Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser,
        Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens,
        George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith,
        Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes,
        Jeffrey Dean. (2016) Google’s Neural Machine Translation System:
        Bridging the Gap between Human and Machine Translation.
        eprint arXiv:1609.08144. https://arxiv.org/pdf/1609.08144v2.pdf
        Retrieved on 27 Oct 2016.

    From Wu et al. (2016):
        "The BLEU score has some undesirable properties when used for single
         sentences, as it was designed to be a corpus measure. We therefore
         use a slightly different score for our RL experiments which we call
         the 'GLEU score'. For the GLEU score, we record all sub-sequences of
         1, 2, 3 or 4 tokens in output and target sequence (n-grams). We then
         compute a recall, which is the ratio of the number of matching n-grams
         to the number of total n-grams in the target (ground truth) sequence,
         and a precision, which is the ratio of the number of matching n-grams
         to the number of total n-grams in the generated output sequence. Then
         GLEU score is simply the minimum of recall and precision. This GLEU
         score's range is always between 0 (no matches) and 1 (all match) and
         it is symmetrical when switching output and target. According to
         our experiments, GLEU score correlates quite well with the BLEU
         metric on a corpus level but does not have its drawbacks for our per
         sentence reward objective."

    Note: The initial implementation only allowed a single reference, but now
          a list of references is required (which is consistent with
          bleu_score.sentence_bleu()).

    The infamous "the the the ... " example

        >>> ref = 'the cat is on the mat'.split()
        >>> hyp = 'the the the the the the the'.split()
        >>> sentence_gleu([ref], hyp)  # doctest: +ELLIPSIS
        0.0909...

    An example to evaluate normal machine translation outputs

        >>> ref1 = str('It is a guide to action that ensures that the military '
        ...            'will forever heed Party commands').split()
        >>> hyp1 = str('It is a guide to action which ensures that the military '
        ...            'always obeys the commands of the party').split()
        >>> hyp2 = str('It is to insure the troops forever hearing the activity '
        ...            'guidebook that party direct').split()
        >>> sentence_gleu([ref1], hyp1) # doctest: +ELLIPSIS
        0.4393...
        >>> sentence_gleu([ref1], hyp2) # doctest: +ELLIPSIS
        0.1206...

    :param references: a list of reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :return: the sentence level GLEU score.
    :rtype: float
    """
    return corpus_gleu([references], [hypothesis], min_len=min_len, max_len=max_len)


def corpus_gleu(
    list_of_references,
    hypotheses,
    min_len=1,
    max_len=4,
    precise_tie_break=False,
    **kwargs,
):
    """
    Calculate a single corpus-level GLEU score (aka. system-level GLEU) for all
    the hypotheses and their respective references.

    Instead of averaging the sentence level GLEU scores (i.e. macro-average
    precision), Wu et al. (2016) sum up the matching tokens and the max of
    hypothesis and reference tokens for each sentence, then compute using the
    aggregate values.

    From Mike Schuster (via email):
        "For the corpus, we just add up the two statistics n_match and
         n_all = max(n_all_output, n_all_target) for all sentences, then
         calculate gleu_score = n_match / n_all, so it is not just a mean of
         the sentence gleu scores (in our case, longer sentences count more,
         which I think makes sense as they are more difficult to translate)."

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']
    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
    ...          'because', 'he', 'read', 'the', 'book']

    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    >>> hypotheses = [hyp1, hyp2]
    >>> corpus_gleu(list_of_references, hypotheses) # doctest: +ELLIPSIS
    0.5673...

    The example below show that corpus_gleu() is different from averaging
    sentence_gleu() for hypotheses

    >>> score1 = sentence_gleu([ref1a], hyp1)
    >>> score2 = sentence_gleu([ref2a], hyp2)
    >>> (score1 + score2) / 2 # doctest: +ELLIPSIS
    0.6144...

    By default, when two references yield the same GLEU ratio, the reference
    encountered first while traversing ``references`` is kept (because
    ``max()`` returns the first maximal element), so the corpus score may
    depend on the order in which the references are listed. Pass
    ``precise_tie_break=True`` to use an exact, order-independent rule: the
    ratios ``n_match / n_all`` are compared exactly on the integer counts and,
    on an exact tie, the reference with the larger denominator ``n_all`` is
    selected. The per-sentence ``(n_match, n_all)`` counts that were selected
    are then returned together with the corpus score. The corpus score itself
    is always the aggregate ``sum(n_match) / sum(n_all)`` over those selected
    per-sentence counts.

    Unigram-only example where the first sentence has two tied references
    (contributing 1/2 and 2/4) and the second sentence contributes 2/2:

        >>> hyp1 = ['a', 'b']
        >>> ref1a = ['a', 'x']               # 1 match out of 2 -> 1/2
        >>> ref1b = ['a', 'b', 'x', 'y']     # 2 matches out of 4 -> 2/4
        >>> hyp2 = ['a', 'b']
        >>> ref2 = ['a', 'b']                # 2 matches out of 2 -> 2/2
        >>> refs = [[ref1a, ref1b], [ref2]]
        >>> hyps = [hyp1, hyp2]
        >>> score, sentence_counts = corpus_gleu(
        ...     refs, hyps, max_len=1, precise_tie_break=True
        ... )
        >>> score  # doctest: +ELLIPSIS
        0.6666...
        >>> sentence_counts
        [(2, 4), (2, 2)]

    Each sentence keeps its own raw counts, so the corpus total is 4/6.
    Flipping the order of the first sentence's references selects the same
    counts and therefore gives the same corpus score:

        >>> flipped_refs = [[ref1b, ref1a], [ref2]]
        >>> score_flipped, counts_flipped = corpus_gleu(
        ...     flipped_refs, hyps, max_len=1, precise_tie_break=True
        ... )
        >>> score_flipped == score
        True
        >>> counts_flipped == sentence_counts
        True

    Without the new rule the original interface is preserved: a bare float is
    returned, computed with the original reference traversal, which is still
    sensitive to reference order on ties:

        >>> corpus_gleu(refs, hyps, max_len=1)  # doctest: +ELLIPSIS
        0.75
        >>> corpus_gleu(flipped_refs, hyps, max_len=1)  # doctest: +ELLIPSIS
        0.6666...

    :param list_of_references: a list of reference sentences, w.r.t. hypotheses
    :type list_of_references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :param precise_tie_break: When ``False`` (default), keep the original
        reference traversal and return only the corpus-level GLEU score.
        When ``True``, compare the per-reference GLEU ratios exactly using the
        integer match counts, break exact ties by choosing the reference with
        the larger denominator, and return the selected per-sentence counts as
        well.
    :type precise_tie_break: bool
    :return: The corpus-level GLEU score, or a ``(score, sentence_counts)``
        tuple when ``precise_tie_break=True``, where ``sentence_counts`` is a
        list of ``(n_match, n_all)`` tuples holding the raw counts selected
        for each hypothesis (``(0, 0)`` for sentences without any usable
        reference).
    :rtype: float or tuple(float, list(tuple(int, int)))
    """
    # ``exact_tie_break`` is accepted as an alias for ``precise_tie_break``.
    precise_tie_break = bool(
        precise_tie_break or kwargs.pop("exact_tie_break", False)
    )
    if kwargs:
        raise TypeError(
            "Unexpected keyword argument(s): {}".format(
                ", ".join(sorted(kwargs))
            )
        )

    # sanity check
    assert len(list_of_references) == len(
        hypotheses
    ), "The number of hypotheses and their reference(s) should be the same"

    # sum matches and max-token-lengths over all sentences
    corpus_n_match = 0
    corpus_n_all = 0

    # The raw (n_match, n_all) counts selected for each hypothesis. Only
    # collected when the exact tie-breaking rule is requested so that the
    # original single-float return interface stays untouched otherwise.
    sentence_counts = []

    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_ngrams = Counter(everygrams(hypothesis, min_len, max_len))
        tpfp = sum(hyp_ngrams.values())  # True positives + False positives.

        hyp_counts = []
        for reference in references:
            ref_ngrams = Counter(everygrams(reference, min_len, max_len))
            tpfn = sum(ref_ngrams.values())  # True positives + False negatives.

            overlap_ngrams = ref_ngrams & hyp_ngrams
            tp = sum(overlap_ngrams.values())  # True positives.

            # While GLEU is defined as the minimum of precision and
            # recall, we can reduce the number of division operations by one by
            # instead finding the maximum of the denominators for the precision
            # and recall formulae, since the numerators are the same:
            #     precision = tp / tpfp
            #     recall = tp / tpfn
            #     gleu_score = min(precision, recall) == tp / max(tpfp, tpfn)
            n_all = max(tpfp, tpfn)

            if n_all > 0:
                hyp_counts.append((tp, n_all))

        # use the reference yielding the highest score
        if hyp_counts:
            if precise_tie_break:
                # Compare the ratios n_match / n_all exactly on the integer
                # counts via cross multiplication (no floating-point
                # rounding). On an exact tie keep the reference with the
                # larger denominator, which makes the choice independent of
                # the order in which the references are listed.
                n_match, n_all = hyp_counts[0]
                for cand_match, cand_all in hyp_counts[1:]:
                    lhs = cand_match * n_all
                    rhs = n_match * cand_all
                    if lhs > rhs or (lhs == rhs and cand_all > n_all):
                        n_match, n_all = cand_match, cand_all
            else:
                n_match, n_all = max(hyp_counts, key=lambda hc: hc[0] / hc[1])
            corpus_n_match += n_match
            corpus_n_all += n_all
            if precise_tie_break:
                sentence_counts.append((n_match, n_all))
        elif precise_tie_break:
            # No usable reference for this sentence: it contributes nothing
            # to the aggregate but still gets an entry in the reported counts.
            sentence_counts.append((0, 0))

    # corner case: empty corpus or empty references---don't divide by zero!
    if corpus_n_all == 0:
        gleu_score = 0.0
    else:
        gleu_score = corpus_n_match / corpus_n_all

    if precise_tie_break:
        return gleu_score, sentence_counts
    return gleu_score
