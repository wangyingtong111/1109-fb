"""Test the module SMOTE."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
import pytest
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn.over_sampling import SMOTE

RND_SEED = 0
X = np.array(
    [
        [0.11622591, -0.0317206],
        [0.77481731, 0.60935141],
        [1.25192108, -0.22367336],
        [0.53366841, -0.30312976],
        [1.52091956, -0.49283504],
        [-0.28162401, -2.10400981],
        [0.83680821, 1.72827342],
        [0.3084254, 0.33299982],
        [0.70472253, -0.73309052],
        [0.28893132, -0.38761769],
        [1.15514042, 0.0129463],
        [0.88407872, 0.35454207],
        [1.31301027, -0.92648734],
        [-1.11515198, -0.93689695],
        [-0.18410027, -0.45194484],
        [0.9281014, 0.53085498],
        [-0.14374509, 0.27370049],
        [-0.41635887, -0.38299653],
        [0.08711622, 0.93259929],
        [1.70580611, -0.11219234],
    ]
)
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_sample_regular():
    smote = SMOTE(random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
            [0.29307743, -0.14670439],
            [0.84976473, -0.15570176],
            [0.61319159, -0.11571668],
            [0.66052536, -0.28246517],
        ]
    )
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    sampling_strategy = {0: 9, 1: 12}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
            [0.36784496, -0.1953161],
        ]
    )
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_with_nn():
    nn_k = NearestNeighbors(n_neighbors=6)
    smote = SMOTE(random_state=RND_SEED, k_neighbors=nn_k)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
            [0.29307743, -0.14670439],
            [0.84976473, -0.15570176],
            [0.61319159, -0.11571668],
            [0.66052536, -0.28246517],
        ]
    )
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


# Small training set: a majority class with six samples, a class with two
# samples (4 and 6) and a singleton class (1).
X_SMALL = np.array(
    [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [4.0], [6.0], [1.0]]
)
Y_SMALL = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2])
SAMPLING_STRATEGY_SMALL = {1: 4, 2: 3}


def test_smote_small_class_strict_raises():
    """Check that the strict strategy keeps the historical nearest-neighbor
    check and raises when a class has too few samples."""
    smote = SMOTE(
        k_neighbors=3,
        sampling_strategy=SAMPLING_STRATEGY_SMALL,
        random_state=RND_SEED,
    )
    # "strict" is the default strategy
    assert smote.small_class_strategy == "strict"
    with pytest.raises(ValueError, match="n_neighbors = 4"):
        smote.fit_resample(X_SMALL, Y_SMALL)


def test_smote_small_class_reduce_without_singleton():
    """Check that the number of neighbors is reduced for a two-sample class."""
    X = X_SMALL[:8]
    y = Y_SMALL[:8]
    smote = SMOTE(
        k_neighbors=3,
        sampling_strategy={1: 4},
        random_state=42,
        small_class_strategy="reduce",
    )
    X_resampled, y_resampled = smote.fit_resample(X, y)

    assert_array_equal(np.bincount(y_resampled), [6, 4])
    class_1_values = X_resampled[y_resampled == 1].ravel()
    # every synthetic sample lies between the two original samples
    assert np.all(class_1_values >= 4.0)
    assert np.all(class_1_values <= 6.0)
    # the two new samples are generated with a fixed random state
    assert_allclose(
        np.sort(class_1_values), [4.0, 4.53601212, 5.90142861, 6.0], rtol=R_TOL
    )
    # the configured estimator is left untouched
    assert smote.nn_k_.n_neighbors == 4


def test_smote_small_class_reduce_singleton_raises():
    """Check that reducing the neighborhood is impossible for a singleton."""
    smote = SMOTE(
        k_neighbors=3,
        sampling_strategy=SAMPLING_STRATEGY_SMALL,
        random_state=RND_SEED,
        small_class_strategy="reduce",
    )
    with pytest.raises(ValueError, match="only one sample"):
        smote.fit_resample(X_SMALL, Y_SMALL)


def test_smote_small_class_duplicate():
    """Check the duplicate strategy: interpolation when possible and
    duplication for the singleton class."""
    smote = SMOTE(
        k_neighbors=3,
        sampling_strategy=SAMPLING_STRATEGY_SMALL,
        random_state=42,
        small_class_strategy="duplicate",
    )
    X_resampled, y_resampled = smote.fit_resample(X_SMALL, Y_SMALL)

    assert_array_equal(np.bincount(y_resampled), [6, 4, 3])
    # the majority class is kept untouched
    assert_allclose(X_resampled[y_resampled == 0].ravel(), X_SMALL[:6].ravel())
    # class 1 keeps its two samples and gets two interpolated samples
    class_1_values = X_resampled[y_resampled == 1].ravel()
    assert np.all(class_1_values >= 4.0)
    assert np.all(class_1_values <= 6.0)
    assert_allclose(
        np.sort(class_1_values), [4.0, 4.53601212, 5.90142861, 6.0], rtol=R_TOL
    )
    # the singleton class is completed with exact copies of its unique sample
    assert_array_equal(X_resampled[y_resampled == 2].ravel(), np.ones(3))


def test_smote_small_class_duplicate_sparse_matches_dense():
    """Check that a sparse input leads to the same resampled content."""
    smote = SMOTE(
        k_neighbors=3,
        sampling_strategy=SAMPLING_STRATEGY_SMALL,
        random_state=42,
        small_class_strategy="duplicate",
    )
    X_dense, y_dense = smote.fit_resample(X_SMALL, Y_SMALL)

    X_sparse, y_sparse = smote.fit_resample(sparse.csr_matrix(X_SMALL), Y_SMALL)

    assert sparse.issparse(X_sparse)
    assert_allclose(X_sparse.toarray(), X_dense)
    assert_array_equal(y_sparse, y_dense)


def test_smote_small_class_duplicate_deterministic():
    """Check that the generation is reproducible with a fixed random state."""
    smote = SMOTE(
        k_neighbors=3,
        sampling_strategy=SAMPLING_STRATEGY_SMALL,
        random_state=42,
        small_class_strategy="duplicate",
    )
    X_first, y_first = smote.fit_resample(X_SMALL, Y_SMALL)
    X_second, y_second = smote.fit_resample(X_SMALL, Y_SMALL)

    assert_allclose(X_first, X_second)
    assert_array_equal(y_first, y_second)


def test_smote_small_class_strategy_with_neighbor_object():
    """Check that the reduced neighborhood also works with a custom nearest
    neighbors estimator."""
    smote = SMOTE(
        k_neighbors=NearestNeighbors(n_neighbors=4),
        sampling_strategy={1: 4},
        random_state=RND_SEED,
        small_class_strategy="reduce",
    )
    X_resampled, y_resampled = smote.fit_resample(X_SMALL[:8], Y_SMALL[:8])

    assert_array_equal(np.bincount(y_resampled), [6, 4])
    assert np.all(X_resampled[y_resampled == 1] >= 4.0)
    assert np.all(X_resampled[y_resampled == 1] <= 6.0)


@pytest.mark.parametrize("strategy", ["strict", "reduce", "duplicate"])
def test_smote_small_class_strategy_normal_data_unchanged(strategy):
    """Check that the strategies do not change the output when the classes are
    large enough."""
    smote = SMOTE(random_state=RND_SEED, small_class_strategy=strategy)
    X_resampled, y_resampled = smote.fit_resample(X, Y)

    assert_allclose(X_resampled[: X.shape[0]], X)
    assert_array_equal(np.bincount(y_resampled), [12, 12])


def test_smote_small_class_strategy_invalid_value():
    """Check that an invalid strategy raises a parameter validation error."""
    with pytest.raises(ValueError, match="small_class_strategy"):
        SMOTE(small_class_strategy="bogus").fit_resample(X, Y)
