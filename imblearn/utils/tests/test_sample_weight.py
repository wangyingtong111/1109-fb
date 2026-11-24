"""Tests for the sample weight propagation utilities."""

# Authors: The imbalanced-learn developers
# License: MIT

import numpy as np
import pytest

from imblearn.utils._sample_weight import (
    check_sample_weight,
    redistribute_sample_weight,
)


def test_check_sample_weight_none_passthrough():
    assert check_sample_weight(None, 4) is None


def test_check_sample_weight_valid():
    weight = check_sample_weight([1.0, 2.0, 3.0, 4.0], 4)
    assert weight.dtype == np.float64
    np.testing.assert_array_equal(weight, [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "sample_weight, match",
    [
        (np.ones((4, 1)), "1-dimensional"),
        (np.ones(3), "inconsistent numbers of samples"),
    ],
)
def test_check_sample_weight_invalid(sample_weight, match):
    with pytest.raises(ValueError, match=match):
        check_sample_weight(sample_weight, 4, estimator_name="DummySampler")


def test_redistribute_sample_weight_splits_copies_equally():
    sample_weight = np.array([1.0, 2.0, 3.0, 6.0])
    # source samples 0, 1, 2 appear once and source sample 3 three times
    sample_indices = np.array([0, 1, 2, 3, 3, 3])

    resampled_weight = redistribute_sample_weight(
        sample_weight=sample_weight,
        sample_indices=sample_indices,
        n_samples=4,
    )

    np.testing.assert_allclose(
        resampled_weight, [1.0, 2.0, 3.0, 2.0, 2.0, 2.0]
    )
    # total weight associated with each source sample is preserved
    for source_idx in range(4):
        mask = sample_indices == source_idx
        assert resampled_weight[mask].sum() == sample_weight[source_idx]


def test_redistribute_sample_weight_under_sampling_shape():
    sample_weight = np.array([1.0, 2.0, 3.0, 6.0])
    sample_indices = np.array([1, 3])

    resampled_weight = redistribute_sample_weight(
        sample_weight=sample_weight,
        sample_indices=sample_indices,
        n_samples=4,
    )

    np.testing.assert_allclose(resampled_weight, [2.0, 6.0])
