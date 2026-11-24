from __future__ import annotations

import pytest

from rapidfuzz import fuzz, process_py
from rapidfuzz.distance import Levenshtein

np = pytest.importorskip("numpy")


def reject_hidden(value):
    return None if value == "hidden" else value


@pytest.mark.parametrize(
    ("scorer", "cutoff", "score"),
    [
        (Levenshtein.distance, 0, 0),
        (Levenshtein.normalized_similarity, 1, 1.0),
        (fuzz.ratio, 100, 100.0),
    ],
)
def test_processor_none_rejects_but_empty_string_remains_comparable(scorer, cutoff, score):
    choices = {"omit": "hidden", "legal": ""}

    assert process_py.extractOne(
        "", choices, processor=reject_hidden, scorer=scorer, score_cutoff=cutoff
    ) == ("", score, "legal")
    assert process_py.extract(
        "", choices, processor=reject_hidden, scorer=scorer, score_cutoff=cutoff, limit=None
    ) == [("", score, "legal")]
    assert list(
        process_py.extract_iter(
            "", choices, processor=reject_hidden, scorer=scorer, score_cutoff=cutoff
        )
    ) == [("", score, "legal")]


def test_empty_string_without_rejection_still_matches_at_zero_cutoff():
    result = process_py.extract(
        "",
        {"legal": "", "other": "x"},
        scorer=Levenshtein.distance,
        score_cutoff=0,
    )
    assert result == [("", 0, "legal")]


def test_processor_returned_empty_string_is_not_rejected():
    result = process_py.extractOne(
        "",
        {"blanked": "hidden-by-processor"},
        processor=lambda value: "" if value == "hidden-by-processor" else value,
        scorer=Levenshtein.distance,
        score_cutoff=0,
    )

    assert result == ("hidden-by-processor", 0, "blanked")


def test_processor_exception_is_not_treated_as_rejection():
    def failing_processor(value):
        if value == "hidden":
            raise ValueError("processor failed")
        return value

    with pytest.raises(ValueError, match="processor failed"):
        process_py.extractOne("", ["hidden", ""], processor=failing_processor)

    with pytest.raises(ValueError, match="processor failed"):
        process_py.cdist([""], ["hidden", ""], processor=failing_processor)


def test_cdist_keeps_rejected_position_as_nan():
    matrix = process_py.cdist(
        [""],
        ["hidden", "", "x"],
        processor=reject_hidden,
        scorer=Levenshtein.distance,
    )

    assert matrix.shape == (1, 3)
    assert matrix.dtype.kind == "f"
    assert np.isnan(matrix[0, 0])
    assert matrix[0, 1] == 0
    assert matrix[0, 2] == 1


def test_cdist_object_dtype_uses_none_for_rejected_position():
    matrix = process_py.cdist(
        [""],
        ["hidden", ""],
        processor=reject_hidden,
        scorer=Levenshtein.distance,
        dtype=object,
    )

    assert matrix[0, 0] is None
    assert matrix[0, 1] == 0
