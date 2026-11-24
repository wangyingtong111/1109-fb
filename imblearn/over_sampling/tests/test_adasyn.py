"""Test the module under sampler."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn.over_sampling import ADASYN

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


def test_ada_init():
    sampling_strategy = "auto"
    ada = ADASYN(sampling_strategy=sampling_strategy, random_state=RND_SEED)
    assert ada.random_state == RND_SEED


def test_ada_fit_resample():
    ada = ADASYN(random_state=RND_SEED)
    X_resampled, y_resampled = ada.fit_resample(X, Y)
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
            [0.88161986, -0.2829741],
            [0.35681689, -0.18814597],
            [1.4148276, 0.05308106],
            [0.3136591, -0.31327875],
        ]
    )
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_fit_resample_nn_obj():
    nn = NearestNeighbors(n_neighbors=6)
    ada = ADASYN(random_state=RND_SEED, n_neighbors=nn)
    X_resampled, y_resampled = ada.fit_resample(X, Y)
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
            [0.88161986, -0.2829741],
            [0.35681689, -0.18814597],
            [1.4148276, 0.05308106],
            [0.3136591, -0.31327875],
        ]
    )
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_generates_exact_number_of_samples():
    """Check that the per-sample quotas sum to the requested sample count.

    Independent rounding of each quota used to either drop samples (all
    quotas rounded down to zero) or create extra samples. The allocation
    must rely on a largest-remainder scheme so that the target of the
    sampling strategy is treated as the true number of samples to
    generate.
    """
    # Three minority samples at 0, 10 and 20 with a single majority
    # neighbour each (0.1, 10.1 and 20.1) and two additional majority
    # samples far away. With ``n_neighbors=1`` every minority sample has
    # the same difficulty (ratio of 1 / 3).
    X = np.array(
        [[0.0], [10.0], [20.0], [0.1], [10.1], [20.1], [100.0], [101.0]]
    )
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])

    majority_X = X[y == 0]
    for n_new_samples in (1, 2):
        ada = ADASYN(
            sampling_strategy={1: 3 + n_new_samples},
            n_neighbors=1,
            random_state=RND_SEED,
        )
        X_resampled, y_resampled = ada.fit_resample(X, y)

        # exactly the requested number of minority samples is created
        assert Counter(y_resampled.tolist()) == {
            0: 5,
            1: 3 + n_new_samples,
        }
        assert len(X_resampled) == len(X) + n_new_samples
        # the five original majority samples are left untouched
        assert_array_equal(
            np.sort(X_resampled[y_resampled == 0].ravel()),
            np.sort(majority_X.ravel()),
        )
        # the original minority samples are kept as well
        minority_values = X_resampled[y_resampled == 1].ravel()
        for minority_sample in X[y == 1].ravel():
            assert minority_sample in minority_values
        # with k=1 the single neighbour of a minority sample is another
        # minority sample, so every synthetic sample lies between two of
        # the original minority samples
        X_new = X_resampled[len(X):]
        assert np.all(X_new >= 0.0)
        assert np.all(X_new <= 20.0)


def test_ada_allocation_sums_to_target_and_is_reproducible():
    """Largest-remainder allocation keeps the exact total on any dataset."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(40, 2))
    y = np.array([0] * 30 + [1] * 10)

    for n_new_samples in range(1, 12):
        ada = ADASYN(
            sampling_strategy={1: 10 + n_new_samples},
            n_neighbors=3,
            random_state=RND_SEED,
        )
        X_resampled, y_resampled = ada.fit_resample(X, y)
        assert np.sum(y_resampled == 1) == 10 + n_new_samples
        assert len(X_resampled) == 40 + n_new_samples

        # a fixed random state makes the interpolation reproducible
        ada_bis = ADASYN(
            sampling_strategy={1: 10 + n_new_samples},
            n_neighbors=3,
            random_state=RND_SEED,
        )
        X_bis, y_bis = ada_bis.fit_resample(X, y)
        assert_allclose(X_resampled, X_bis)
        assert_array_equal(y_resampled, y_bis)
