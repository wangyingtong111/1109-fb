"""Utilities to propagate sample weights through duplication-based samplers."""

# Authors: The imbalanced-learn developers
# License: MIT

import numpy as np


def check_sample_weight(sample_weight, n_samples, *, estimator_name="estimator"):
    """Validate that ``sample_weight`` is aligned with the number of samples.

    Parameters
    ----------
    sample_weight : array-like of shape (n_samples,) or None
        Weight of each sample. When ``None``, no validation is performed.

    n_samples : int
        Number of samples expected.

    estimator_name : str, default="estimator"
        Name of the estimator used in the error message.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,) or None
        The validated sample weights as a 1-dimensional numpy array, or
        ``None`` if ``sample_weight`` was ``None``.
    """
    if sample_weight is None:
        return None
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if sample_weight.ndim != 1:
        raise ValueError(
            f"{estimator_name} only supports 1-dimensional `sample_weight`. "
            f"Got an array with {sample_weight.ndim} dimensions instead."
        )
    if sample_weight.shape[0] != n_samples:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: the "
            f"input data has {n_samples} samples while `sample_weight` has "
            f"{sample_weight.shape[0]} elements."
        )
    return sample_weight


def redistribute_sample_weight(sample_weight, sample_indices, n_samples):
    """Redistribute source sample weights onto the resampled rows.

    Each resampled row points to a source sample through ``sample_indices``.
    The weight of a source sample is split equally between all the resampled
    rows derived from it, so that the total weight associated with each
    source sample is preserved.

    Parameters
    ----------
    sample_weight : ndarray of shape (n_samples,)
        Weight of each source sample.

    sample_indices : ndarray of shape (n_resampled_samples,)
        Index of the source sample used for each row of the resampled
        dataset.

    n_samples : int
        Number of source samples.

    Returns
    -------
    sample_weight_resampled : ndarray of shape (n_resampled_samples,)
        Weight associated with each row of the resampled dataset.
    """
    occurrences = np.bincount(sample_indices, minlength=n_samples)
    return sample_weight[sample_indices] / occurrences[sample_indices]
