#
# Diagnostic: can the phrase-level error dictionary produced by
# jiwer.alignment.collect_error_counts be used to recover the word-level
# WER numerator (substitutions + deletions + insertions)?
#
# This script is READ-ONLY with respect to jiwer: it only calls the existing
# public word-level interface
#       jiwer.process_words(...)
# and the existing error-aggregation interface
#       jiwer.collect_error_counts(output)
# It does not modify any metric, dataclass, or report/visualization format.
#
# Run:
#   .venv/Scripts/python.exe diagnose_phrase_error_dict.py
#
import re

import jiwer
from jiwer import transforms as tr

REFERENCE = "red blue and green"
HYPOTHESIS = "tan teal and green"
# Phrases that must be treated as ONE token; "and" and "green" stay separate.
PHRASES = ["red blue", "tan teal"]


# --------------------------------------------------------------------------- #
# Custom transform (built only from the public AbstractTransform building block)
# --------------------------------------------------------------------------- #
class GroupPhrases(tr.AbstractTransform):
    """Tokenize a sentence, treating each multi-word phrase as a single token.

    Every other word remains its own token, so the output is still a
    list[list[str]] as required by jiwer.process_words.
    """

    def __init__(self, phrases):
        # longest first, and escape for regex; match phrases on word boundaries
        self._phrases = sorted(phrases, key=len, reverse=True)
        alternatives = "|".join(re.escape(p) for p in self._phrases)
        self._pattern = re.compile(r"\b(?:%s)\b|\S+" % alternatives)

    def process_string(self, s):
        return self._pattern.findall(s)

    def process_list(self, inp):
        return [self.process_string(s) for s in inp]


phrase_transform = GroupPhrases(PHRASES)


# --------------------------------------------------------------------------- #
# Helpers (pure diagnostics; they do not touch jiwer internals)
# --------------------------------------------------------------------------- #
def show(title, output):
    print("=" * 72)
    print(title)
    print("-" * 72)
    print("reference tokens :", output.references[0])
    print("hypothesis tokens:", output.hypotheses[0])

    print("half-open alignment intervals (type, ref[i1:i2), hyp[j1:j2)):")
    for chunk in output.alignments[0]:
        ref_span = output.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
        hyp_span = output.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]
        print(
            "  %-10s ref[%d:%d)=%r  hyp[%d:%d)=%r"
            % (
                chunk.type,
                chunk.ref_start_idx,
                chunk.ref_end_idx,
                ref_span,
                chunk.hyp_start_idx,
                chunk.hyp_end_idx,
                hyp_span,
            )
        )

    subs, ins, dels = jiwer.collect_error_counts(output)
    print("phrase error dictionary from collect_error_counts():")
    print("  substitutions:", dict(subs))
    print("  insertions   :", dict(ins))
    print("  deletions    :", dict(dels))

    print(
        "jiwer totals: S=%d D=%d I=%d H=%d  ->  WER numerator (S+D+I) = %d"
        % (
            output.substitutions,
            output.deletions,
            output.insertions,
            output.hits,
            output.substitutions + output.deletions + output.insertions,
        )
    )
    return subs, ins, dels


def naive_re_split_counts(subs, ins, dels):
    """What a report-only consumer is forced to do: re-split the text keys.

    Every dictionary entry is counted once and each whitespace-separated piece
    of a key is treated as one edit, regardless of the real chunk lengths.
    """
    s = sum(len(k[0].split()) for k in subs)
    i = sum(len(k.split()) for k in ins)
    d = sum(len(k.split()) for k in dels)
    return s, d, i


def naive_re_split_weighted(subs, ins, dels):
    """Same idea, but honoring the dictionary frequencies (the 'Nx' values)."""
    s = sum(len(k[0].split()) * n for k, n in subs.items())
    i = sum(len(k.split()) * n for k, n in ins.items())
    d = sum(len(k.split()) * n for k, n in dels.items())
    return s, d, i


def counts_from_intervals(output):
    """Re-derive the counts directly from the half-open AlignmentChunks.

    This is exactly the bookkeeping jiwer.process_words performs internally;
    recomputing it here only READS the public alignment output.
    """
    s = d = i = h = 0
    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            h += chunk.ref_end_idx - chunk.ref_start_idx
        elif chunk.type == "substitute":
            s += chunk.ref_end_idx - chunk.ref_start_idx
        elif chunk.type == "delete":
            d += chunk.ref_end_idx - chunk.ref_start_idx
        elif chunk.type == "insert":
            i += chunk.hyp_end_idx - chunk.hyp_start_idx
    return s, d, i, h


# --------------------------------------------------------------------------- #
# Step 0: single-word substitution calibration
# --------------------------------------------------------------------------- #
print("#" * 72)
print("# STEP 0 - CALIBRATION: one single-word substitution anchors the count")
print("#" * 72)
calibration = jiwer.process_words("red and green", "tan and green")
c_subs, c_ins, c_dels = show("single-word substitution (default tokenization)", calibration)
c_naive = naive_re_split_weighted(c_subs, c_ins, c_dels)
c_interval = counts_from_intervals(calibration)
print(
    "calibration check: dict re-split S+D+I = %d, interval S+D+I = %d, jiwer S+D+I = %d"
    % (
        sum(c_naive),
        sum(c_interval[:3]),
        calibration.substitutions
        + calibration.deletions
        + calibration.insertions,
    )
)
print("-> with a single-word key, one dictionary entry == one edit on all three paths.\n")

# --------------------------------------------------------------------------- #
# Step 1: the actual pair under both tokenizations
# --------------------------------------------------------------------------- #
print("#" * 72)
print("# STEP 1 - TARGET PAIR UNDER TWO TOKENIZATIONS")
print("#" * 72)

default_out = jiwer.process_words(REFERENCE, HYPOTHESIS)
d_subs, d_ins, d_dels = show("A) default tokenization (jiwer.wer_default)", default_out)
print(jiwer.visualize_alignment(default_out, show_measures=False))
print()

phrase_out = jiwer.process_words(
    REFERENCE,
    HYPOTHESIS,
    reference_transform=phrase_transform,
    hypothesis_transform=phrase_transform,
)
p_subs, p_ins, p_dels = show("B) phrase transform ('red blue'/'tan teal' = one token)", phrase_out)
print(jiwer.visualize_alignment(phrase_out, show_measures=False))
print()

# --------------------------------------------------------------------------- #
# Step 2: the phrase dictionaries are indistinguishable, the counts are not
# --------------------------------------------------------------------------- #
print("#" * 72)
print("# STEP 2 - WHY RE-SPLITTING THE DICTIONARY TEXT CANNOT RECOVER THE COUNT")
print("#" * 72)

same_dict = (
    dict(d_subs) == dict(p_subs)
    and dict(d_ins) == dict(p_ins)
    and dict(d_dels) == dict(p_dels)
)
print("identical phrase dictionaries in A and B? ", same_dict)
print("  A dict:", dict(d_subs), dict(d_ins), dict(d_dels))
print("  B dict:", dict(p_subs), dict(p_ins), dict(p_dels))

print()
print("word-edit total actually reported by jiwer (S+D+I):")
print("  A default tokens :", default_out.substitutions
      + default_out.deletions + default_out.insertions)
print("  B phrase tokens  :", phrase_out.substitutions
      + phrase_out.deletions + phrase_out.insertions)

print()
print("naive 're-split the text keys' estimate of S+D+I:")
for name, dicts in (("A", (d_subs, d_ins, d_dels)), ("B", (p_subs, p_ins, p_dels))):
    s, d, i = naive_re_split_counts(*dicts)
    sw, dw, iw = naive_re_split_weighted(*dicts)
    print("  %s: unweighted=%d (S=%d D=%d I=%d)   weighted=%d (S=%d D=%d I=%d)"
          % (name, s + d + i, s, d, i, sw + dw + iw, sw, dw, iw))

print()
print("Explanation:")
print("  * collect_error_counts joins every token inside one alignment chunk")
print("    with ' ' into a single string key. In A the substitute chunk spans")
print("    TWO ref tokens and TWO hyp tokens, yet the stored key is just the")
print("    phrase ('red blue','tan teal'); in B that same phrase is ONE token.")
print("  * The key therefore loses the chunk's width. Re-splitting on spaces")
print("    assumes 'one word == one edit', which is true for B (1 edit) but")
print("    false for A (2 substitutions). Both runs hand back the SAME key, so")
print("    a report holding only the dictionary cannot tell the cases apart and")
print("    cannot reconstruct both totals (1 vs 2) from one representation.")
print("  * Re-splitting also silently invents an edit decomposition: an unequal")
print("    chunk (e.g. 2 ref -> 1 hyp) is a block counted once by Levenshtein,")
print("    but space-splitting would turn it into several word edits. Frequency")
print("    values ('Nx') only restore how many chunks occurred, never their size.")

# Supplementary confirmation: even when the words inside a key are countable,
# the dictionary carries no structural information. rapidfuzz moves any length
# imbalance into separate insert/delete blocks, so the edit count cannot be read
# off the joined text; the block boundaries live only in the intervals.
print()
print("supplementary pair (a multi-word substitute block next to an insert block):")
extra = jiwer.process_words(
    "alpha beta green", "gamma delta epsilon green"
)
e_subs, e_ins, e_dels = show("supplementary default-tokenization pair", extra)
sub_key = next(iter(e_subs))
print(
    "substitution key %r is ONE dictionary entry, yet it represents a block of"
    % (sub_key,)
)
print(
    "S=%d (one substitution per reference token); insertion key %r is a "
    "separate block of I=%d"
    % (extra.substitutions, next(iter(e_ins)), extra.insertions)
)
print(
    "jiwer S+D+I = %d. The dictionary does not record the block boundary between"
    % (extra.substitutions + extra.deletions + extra.insertions)
)
print("the insert and the substitute, nor the ref/hyp spans. Re-splitting text")
print("keys can only count words, not reconstruct which block they belonged to.")
print("The interval object instead gives ref[0:2)/hyp[1:3) for the substitute")
print("and ref[0:0)/hyp[0:1) for the insert, so widths and boundaries are exact.\n")

# --------------------------------------------------------------------------- #
# Step 3: the correct way to re-check counts - use the half-open intervals
# --------------------------------------------------------------------------- #
print()
print("#" * 72)
print("# STEP 3 - RE-CHECKING COUNTS FROM THE HALF-OPEN ALIGNMENT INTERVALS")
print("#" * 72)
for name, output in (("A default", default_out), ("B phrase ", phrase_out)):
    s, d, i, h = counts_from_intervals(output)
    jiwer_total = (
        output.substitutions + output.deletions + output.insertions
    )
    ref_len = sum(len(r) for r in output.references)
    assert (s, d, i, h) == (
        output.substitutions,
        output.deletions,
        output.insertions,
        output.hits,
    ), "interval recomputation disagrees with jiwer!"
    print(
        "%s: interval S=%d D=%d I=%d H=%d | S+D+I=%d matches jiwer=%s | "
        "WER=(S+D+I)/len(ref)=%d/%d=%.4f"
        % (
            name,
            s,
            d,
            i,
            h,
            s + d + i,
            (s + d + i) == jiwer_total,
            s + d + i,
            ref_len,
            (s + d + i) / ref_len,
        )
    )

print()
print("Interval rules (end indices are EXCLUSIVE):")
print("  equal      : add ref_end-ref_start to hits")
print("  substitute : add ref_end-ref_start to substitutions")
print("  delete     : add ref_end-ref_start to deletions")
print("  insert     : add hyp_end-hyp_start to insertions")
print("Keep the AlignmentChunks (or these spans) alongside the phrase dictionary;")
print("that interval structure - not the joined text keys - is what reproduces")
print("the word-level WER numerator.")
