"""Class to perform under-sampling by removing Tomek's links."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

import numbers

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from sklearn.utils._param_validation import StrOptions

from imblearn.under_sampling.base import BaseCleaningSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class TomekLinks(BaseCleaningSampler):
    """Under-sampling by removing Tomek's links.

    Read more in the :ref:`User Guide <tomek_links>`.

    Parameters
    ----------
    {sampling_strategy}

    {n_jobs}

    tie_break : {{"single", "all"}}, default="single"
        How to handle samples having several equidistant nearest neighbours.

        - ``"single"``: keeps the historical behaviour. Only a single nearest
          neighbour is considered for every sample. When several neighbours
          are tied at the minimum distance, the one that is returned depends on
          the order of the samples and the result can therefore change when the
          input is reordered.

        - ``"all"``: explicit tie-breaking mode. Every neighbour located at the
          minimum distance is considered as a nearest neighbour. A pair of
          samples from different classes forms a Tomek's link only when each
          sample is a true nearest neighbour of the other, i.e. it belongs to
          the set of minimum-distance neighbours of the other sample. This
          detection is independent of the order of the samples. Only the samples
          from the classes targeted by ``sampling_strategy`` are removed.

        .. versionadded:: 0.15

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    EditedNearestNeighbours : Undersample by samples edition.

    CondensedNearestNeighbour : Undersample by samples condensation.

    RandomUnderSampler : Randomly under-sample the dataset.

    Notes
    -----
    This method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    When ``tie_break="single"`` and several neighbours are equidistant from a
    sample, only the neighbour selected by the nearest-neighbour search is
    inspected. This arbitrary choice makes the detected links sensitive to the
    order of the samples. Setting ``tie_break="all"`` removes this ambiguity by
    requiring that the two samples of a pair are mutual true nearest neighbours
    before considering the pair as a Tomek's link.

    References
    ----------
    .. [1] I. Tomek, "Two modifications of CNN," In Systems, Man, and
       Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 1976.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import TomekLinks
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> tl = TomekLinks()
    >>> X_res, y_res = tl.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 897, 0: 100}})
    """

    _parameter_constraints: dict = {
        **BaseCleaningSampler._parameter_constraints,
        "n_jobs": [numbers.Integral, None],
        "tie_break": [StrOptions({"single", "all"})],
    }

    def __init__(self, *, sampling_strategy="auto", n_jobs=None, tie_break="single"):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_jobs = n_jobs
        self.tie_break = tie_break

    @staticmethod
    def is_tomek(y, nn_index, class_type):
        """Detect if samples are Tomek's link.

        More precisely, it uses the target vector and the first neighbour of
        every sample point and looks for Tomek pairs. Returning a boolean
        vector with True for majority Tomek links.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not.

        nn_index : ndarray of shape (len(y),)
            The index of the closes nearest neighbour to a sample point.

        class_type : int or str
            The label of the minority class.

        Returns
        -------
        is_tomek : ndarray of shape (len(y), )
            Boolean vector on len( # samples ), with True for majority samples
            that are Tomek links.
        """
        links = np.zeros(len(y), dtype=bool)

        # find which class to not consider
        class_excluded = [c for c in np.unique(y) if c not in class_type]

        # there is a Tomek link between two samples if they are both nearest
        # neighbors of each others.
        for index_sample, target_sample in enumerate(y):
            if target_sample in class_excluded:
                continue

            if y[nn_index[index_sample]] != target_sample:
                if nn_index[nn_index[index_sample]] == index_sample:
                    links[index_sample] = True

        return links

    @staticmethod
    def is_tomek_with_ties(y, nn_indices, class_type):
        """Detect Tomek's links while accounting for equidistant neighbours.

        Every index stored in ``nn_indices`` is considered as a nearest
        neighbour of a sample (i.e. it is located at the minimum distance).
        Two samples from different classes form a Tomek's link when each of
        them is a true nearest neighbour of the other. Only the samples from
        the classes targeted by ``class_type`` are flagged for removal.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to a targeted class or not.

        nn_indices : ragged sequence of ndarray
            ``nn_indices[i]`` contains the indices of every nearest neighbour
            (all the samples located at the minimum distance) of sample ``i``.

        class_type : dict, int or str
            The class labels targeted by the resampling. A dictionary keyed by
            the targeted class labels or a single class label are accepted.

        Returns
        -------
        is_tomek : ndarray of shape (n_samples,)
            Boolean vector with True for the targeted samples that belong to a
            Tomek's link.
        """
        links = np.zeros(len(y), dtype=bool)

        if isinstance(class_type, dict):
            targeted_classes = tuple(class_type)
        else:
            targeted_classes = (class_type,)

        for index_sample, target_sample in enumerate(y):
            if target_sample not in targeted_classes:
                continue

            for neighbor_index in nn_indices[index_sample]:
                # a link requires neighbours from different classes and mutual
                # membership in each other's set of minimum-distance neighbours
                if y[neighbor_index] != target_sample and (
                    index_sample in nn_indices[neighbor_index]
                ):
                    links[index_sample] = True
                    break

        return links

    def _fit_resample(self, X, y):
        if self.tie_break == "single":
            # Find the nearest neighbour of every point
            nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
            nn.fit(X)
            nns = nn.kneighbors(X, return_distance=False)[:, 1]

            links = self.is_tomek(y, nns, self.sampling_strategy_)
        else:
            # Explicit tie-breaking mode: consider every neighbour located at
            # the minimum distance instead of picking a single one.
            links = np.zeros(X.shape[0], dtype=bool)

            if X.shape[0] > 1:
                nn = NearestNeighbors(n_jobs=self.n_jobs)
                nn.fit(X)

                # Minimum distance from each sample to another sample.
                distances, _ = nn.kneighbors(X, n_neighbors=2)
                min_distances = distances[:, 1]

                # Gather the samples lying within the largest minimum distance
                # and keep, for each row, only the samples located at its own
                # minimum distance. The query always returns the sample itself
                # (distance zero), which is explicitly discarded using its index
                # so that coincident samples from another class are retained.
                radius = np.max(min_distances)
                radius_distances, radius_neighbors = nn.radius_neighbors(
                    X, radius=radius
                )
                nn_indices = [
                    neighbor_indices[
                        (neighbor_indices != index_sample)
                        & np.isclose(neighbor_distances, min_distance)
                    ]
                    for index_sample, (
                        neighbor_indices,
                        neighbor_distances,
                        min_distance,
                    ) in enumerate(
                        zip(radius_neighbors, radius_distances, min_distances)
                    )
                ]

                links = self.is_tomek_with_ties(
                    y, nn_indices, self.sampling_strategy_
                )

        self.sample_indices_ = np.flatnonzero(np.logical_not(links))

        return (
            _safe_indexing(X, self.sample_indices_),
            _safe_indexing(y, self.sample_indices_),
        )

    def _more_tags(self):
        return {"sample_indices": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.sampler_tags.sample_indices = True
        return tags
