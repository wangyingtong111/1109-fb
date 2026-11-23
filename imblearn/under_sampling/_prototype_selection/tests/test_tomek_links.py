"""Test the module Tomek's links."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal

from imblearn.under_sampling import TomekLinks

X = np.array(
    [
        [0.31230513, 0.1216318],
        [0.68481731, 0.51935141],
        [1.34192108, -0.13367336],
        [0.62366841, -0.21312976],
        [1.61091956, -0.40283504],
        [-0.37162401, -2.19400981],
        [0.74680821, 1.63827342],
        [0.2184254, 0.24299982],
        [0.61472253, -0.82309052],
        [0.19893132, -0.47761769],
        [1.06514042, -0.0770537],
        [0.97407872, 0.44454207],
        [1.40301027, -0.83648734],
        [-1.20515198, -1.02689695],
        [-0.27410027, -0.54194484],
        [0.8381014, 0.44085498],
        [-0.23374509, 0.18370049],
        [-0.32635887, -0.29299653],
        [-0.00288378, 0.84259929],
        [1.79580611, -0.02219234],
    ]
)
Y = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])


def test_tl_init():
    tl = TomekLinks()
    assert tl.n_jobs is None
    assert tl.tie_break == "single"


def test_tomek_links_tie_break_invalid_strings():
    """Check that an invalid `tie_break` raises a parameter error."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="tie_break"):
        TomekLinks(tie_break="invalid").fit_resample(X, y)


def test_tomek_links_tie_break_all_equidistant_neighbours():
    """Check the explicit tie-breaking mode on a boundary with two majority
    samples equidistant from the minority sample.

    The samples 0 and 2 (majority class) both have the sample 1 (minority
    class) as a nearest neighbour, while 1 is equidistant from 0 and 2. The
    tie-breaking mode keeps the minority sample and removes both majority
    samples of the boundary, regardless of the order of the input.
    """
    X = np.array([[0.0], [2.0], [10.0], [1.0]])
    y = np.array([1, 1, 1, 0])

    tl = TomekLinks(tie_break="all")
    X_resampled, y_resampled = tl.fit_resample(X, y)

    assert_array_equal(np.sort(tl.sample_indices_), np.array([2, 3]))
    assert_array_equal(np.sort(X_resampled.ravel()), np.array([1.0, 10.0]))
    assert_array_equal(np.sort(y_resampled), np.array([0, 1]))

    # Swapping the order of the two equidistant majority samples must lead to
    # the same retained set of values.
    permutation = np.array([1, 0, 2, 3])
    tl_permuted = TomekLinks(tie_break="all")
    X_permuted, y_permuted = tl_permuted.fit_resample(X[permutation], y[permutation])

    assert set(X_permuted.ravel()) == set(X_resampled.ravel())
    assert set(y_permuted) == set(y_resampled)


def test_tomek_links_tie_break_all_only_targeted_class_removed():
    """Check that the tie-breaking mode only removes samples from the classes
    targeted by `sampling_strategy`."""
    X = np.array([[0.0], [2.0], [10.0], [1.0]])
    y = np.array([1, 1, 1, 0])

    # By default only the majority class is cleaned: the minority sample at 1
    # is retained even though it belongs to a Tomek's link.
    tl = TomekLinks(tie_break="all")
    X_resampled, y_resampled = tl.fit_resample(X, y)
    assert_array_equal(np.sort(X_resampled.ravel()), np.array([1.0, 10.0]))

    # With `sampling_strategy="all"`, both sides of the links are removed.
    tl_all = TomekLinks(tie_break="all", sampling_strategy="all")
    X_resampled, y_resampled = tl_all.fit_resample(X, y)
    assert_array_equal(X_resampled.ravel(), np.array([10.0]))
    assert_array_equal(y_resampled, np.array([1]))


def test_tomek_links_tie_break_single_legacy_behaviour():
    """Check that the historical single-neighbour behaviour remains available
    and is still sensitive to the order of the samples on a tie."""
    X = np.array([[0.0], [2.0], [10.0], [1.0]])
    y = np.array([1, 1, 1, 0])

    tl = TomekLinks(tie_break="single")
    X_resampled, y_resampled = tl.fit_resample(X, y)

    # A single nearest neighbour is picked for the minority sample, so only one
    # of the two equidistant majority samples is removed.
    assert set(X_resampled.ravel()) == {1.0, 2.0, 10.0}

    permutation = np.array([1, 0, 2, 3])
    tl_permuted = TomekLinks(tie_break="single")
    X_permuted, _ = tl_permuted.fit_resample(X[permutation], y[permutation])
    assert set(X_permuted.ravel()) == {0.0, 1.0, 10.0}


def test_tomek_links_tie_break_matches_single_without_ties():
    """Check that both modes give the same result when no distance tie is
    present."""
    tl_single = TomekLinks(tie_break="single")
    X_single, y_single = tl_single.fit_resample(X, Y)

    tl_all = TomekLinks(tie_break="all")
    X_all, y_all = tl_all.fit_resample(X, Y)

    assert_array_equal(X_single, X_all)
    assert_array_equal(y_single, y_all)
    assert_array_equal(tl_single.sample_indices_, tl_all.sample_indices_)


def test_tomek_links_tie_break_all_two_samples():
    """Check the tie-breaking mode on a minimal dataset with two samples."""
    X = np.array([[0.0], [1.0]])
    y = np.array([1, 0])

    tl = TomekLinks(tie_break="all")
    X_resampled, y_resampled = tl.fit_resample(X, y)
    assert_array_equal(X_resampled.ravel(), np.array([1.0]))
    assert_array_equal(y_resampled, np.array([0]))


def test_tl_fit_resample():
    tl = TomekLinks()
    X_resampled, y_resampled = tl.fit_resample(X, Y)

    X_gt = np.array(
        [
            [0.31230513, 0.1216318],
            [0.68481731, 0.51935141],
            [1.34192108, -0.13367336],
            [0.62366841, -0.21312976],
            [1.61091956, -0.40283504],
            [-0.37162401, -2.19400981],
            [0.74680821, 1.63827342],
            [0.2184254, 0.24299982],
            [0.61472253, -0.82309052],
            [0.19893132, -0.47761769],
            [0.97407872, 0.44454207],
            [1.40301027, -0.83648734],
            [-1.20515198, -1.02689695],
            [-0.23374509, 0.18370049],
            [-0.32635887, -0.29299653],
            [-0.00288378, 0.84259929],
            [1.79580611, -0.02219234],
        ]
    )
    y_gt = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.parametrize(
    "sampling_strategy", ["auto", "majority", "not minority", "not majority", "all"]
)
def test_tomek_links_strings(sampling_strategy):
    """Check that we support all supposed strings as `sampling_strategy` in
    a sampler inheriting from `BaseCleaningSampler`."""

    X, y = make_classification(
        n_samples=100,
        n_clusters_per_class=1,
        n_classes=3,
        weights=[0.1, 0.3, 0.6],
        random_state=0,
    )
    TomekLinks(sampling_strategy=sampling_strategy).fit_resample(X, y)
