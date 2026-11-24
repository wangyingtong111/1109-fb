"""Quantile Encoder."""

from __future__ import annotations

__author__ = 'david26694', 'cmougan'

import operator
import warnings
from functools import reduce
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder


def _weighted_quantiles_of_sorted_groups(
    target: np.ndarray,
    cumulative_weight: np.ndarray,
    group_ends: np.ndarray,
    offsets: np.ndarray,
    totals: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """Quantiles of contiguous, target-sorted groups of frequency-weighted rows.

    The rows are sorted by group and then by ascending target, so a group is the
    range ending at ``group_ends[i]``. ``offsets[i]`` is the cumulative weight
    before the group and ``totals[i]`` its summed weight. The quantile is taken
    from the virtual expanded sample of size ``totals[i]`` with numpy's default
    ('linear') interpolation between the two neighbouring ranks.
    """
    position = quantile * np.maximum(totals - 1.0, 0.0)
    lower = np.floor(position)
    upper = np.ceil(position)
    # The value whose expanded positions [cumulative_weight[i] - weight_i,
    # cumulative_weight[i]) contain the (fractional) rank; capped at the group's
    # last row because searchsorted looks at the whole array.
    lower_index = np.minimum(
        np.searchsorted(cumulative_weight, offsets + lower, 'right'), group_ends
    )
    upper_index = np.minimum(
        np.searchsorted(cumulative_weight, offsets + upper, 'right'), group_ends
    )
    lower_value = target[lower_index]
    upper_value = target[upper_index]
    return lower_value + (position - lower) * (upper_value - lower_value)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Quantile of ``values`` where each value is repeated ``weights`` times.

    Frequency weights are interpreted as an implicit repetition of the rows, so
    for positive integer weights the result equals ``numpy.quantile`` on the
    expanded sample, using the same (linear) interpolation. Zero-weight rows are
    ignored entirely, which keeps them from influencing the result.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    positive = weights > 0
    values = values[positive]
    weights = weights[positive]
    if values.size == 0:
        return np.nan

    order = np.argsort(values, kind='stable')
    values = values[order]
    weights = weights[order]

    cumulative_weight = np.cumsum(weights)
    totals = cumulative_weight[-1:]
    quantiles = _weighted_quantiles_of_sorted_groups(
        values,
        cumulative_weight,
        np.array([values.size - 1]),
        np.zeros(1),
        totals,
        quantile,
    )
    return float(quantiles[0])


def _weighted_group_statistics(
    codes: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> tuple[pd.DataFrame, int]:
    """Per-code quantile/sum/count of the target using frequency weights.

    Only rows with a positive weight are taken into account, so zero-weight rows
    neither contribute to the statistics of a category nor lift the prior. The
    statistics are reindexed to every observed code, so a category that only has
    zero-weight rows is still known but has a count of zero.

    Returns the statistics and the number of codes with a positive weight.
    """
    codes = np.asarray(codes)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # A stats row per observed code, even if all of its rows carry zero weight.
    known = pd.notna(codes)
    all_codes = np.unique(codes[known])

    positive = weights > 0
    codes = codes[known & positive]
    target = target[known & positive]
    weights = weights[known & positive]
    if codes.size == 0:
        empty = pd.DataFrame(
            {'quantile': np.nan, 'sum': 0.0, 'count': 0.0}, index=all_codes, dtype=float
        )
        return empty, 0

    # Sort by code and then by target, so each category is a contiguous run of
    # targets in ascending order.
    order = np.lexsort((target, codes))
    codes, target, weights = codes[order], target[order], weights[order]

    cumulative_weight = np.cumsum(weights)
    cumulative_target = np.cumsum(target * weights)
    group_ends = np.flatnonzero(np.r_[codes[1:] != codes[:-1], True])
    # Totals and offsets of each category, relative to the global cumulative sums.
    previous_ends = np.r_[0, group_ends[:-1] + 1] - 1
    offsets = np.where(previous_ends >= 0, cumulative_weight[previous_ends], 0.0)
    totals = cumulative_weight[group_ends] - offsets

    quantiles = _weighted_quantiles_of_sorted_groups(
        target, cumulative_weight, group_ends, offsets, totals, quantile
    )

    total_of_previous = np.where(previous_ends >= 0, cumulative_target[previous_ends], 0.0)
    stats = pd.DataFrame(
        {
            'quantile': quantiles,
            'sum': cumulative_target[group_ends] - total_of_previous,
            'count': totals,
        },
        index=codes[group_ends],
    )
    stats = stats.reindex(all_codes)
    return stats[['quantile', 'sum', 'count']], len(group_ends)


def _validate_sample_weight(sample_weight, n_samples: int) -> np.ndarray:
    """Validate frequency weights passed to ``fit`` and return them as an array.

    Weights are taken positionally (they are not aligned on the pandas index),
    matching scikit-learn's ``sample_weight`` convention.
    """
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.shape[0] != n_samples:
        raise ValueError(
            f'sample_weight has {weights.shape[0]} entries but the target has {n_samples} rows.'
        )
    if not np.isfinite(weights).all():
        raise ValueError('sample_weight must contain only finite values.')
    if (weights < 0).any():
        raise ValueError('sample_weight cannot contain negative values.')
    if not (weights > 0).any():
        raise ValueError('sample_weight must contain at least one positive value.')
    return weights


class QuantileEncoder(util.SupervisedTransformerMixin, util.BaseEncoder):
    """Quantile Encoding for categorical features.

    This a statistically modified version of target MEstimate encoder where selected features
    are replaced by the statistical quantile instead of the mean. Replacing with the
    median is a particular case where self.quantile = 0.5. In comparison to MEstimateEncoder
    it has two tunable parameter `m` and `quantile`

    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    quantile: float
        float indicating statistical quantile. ´0.5´ for median.
    m: float
        this is the “m” in the m-probability estimate. Higher value of m results into
        stronger shrinking. M is non-negative. 0 for no smoothing.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform
        (otherwise it will be a numpy array).
    handle_missing: str, int, float or callable
        options are 'error', 'return_nan'  and 'value', defaults to 'value',
        which returns the target quantile. A number is used as the encoded value
        for missing values that were not seen at fit time, and a callable
        fn(value, mapping) is evaluated once per column when the mapping is
        finalized during fit.
    handle_unknown: str, int, float or callable
        options are 'error', 'return_nan' and 'value', defaults to 'value',
        which returns the target quantile. A number is used as the encoded value
        for unseen categories, and a callable fn(value, mapping) is evaluated
        once per column when the mapping is finalized during fit.

    Notes
    -----
    Columns in which every category is unique (id-like columns) are not
    encoded with their own category statistics: all their categories are
    encoded with the prior quantile instead, which prevents overfitting on
    identifier-like columns.

    Frequency weights can be passed as ``sample_weight`` to ``fit`` (and
    ``fit_transform``). A row with weight ``w`` then counts as ``w`` copies of
    that row, so the data does not have to be materialized row by row. The
    per-category quantiles, the global prior quantile used for unknown/missing
    categories, and the counts used for the ``m`` smoothing are all computed
    from that same weighted data; rows with zero weight are ignored. For
    integer weights this reproduces fitting on the expanded data, using the
    same linear interpolation as :func:`numpy.quantile`. Without
    ``sample_weight`` the behaviour is unchanged.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name='house_prices', as_frame=True)
    >>> display_cols = [
    ...     'Id',
    ...     'MSSubClass',
    ...     'MSZoning',
    ...     'LotFrontage',
    ...     'YearBuilt',
    ...     'Heating',
    ...     'CentralAir',
    ... ]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = QuantileEncoder(cols=['CentralAir', 'Heating'], quantile=0.5, m=1.0).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 7 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   Id           1460 non-null   float64
     1   MSSubClass   1460 non-null   float64
     2   MSZoning     1460 non-null   object
     3   LotFrontage  1201 non-null   float64
     4   YearBuilt    1460 non-null   float64
     5   Heating      1460 non-null   float64
     6   CentralAir   1460 non-null   float64
    dtypes: float64(6), object(1)
    memory usage: 80.0+ KB
    None

    References
    ----------

    .. [1] Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems,
        https://link.springer.com/chapter/10.1007%2F978-3-030-85529-1_14
    .. [2] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification
        and Prediction Problems, equation 7, from https://dl.acm.org/citation.cfm?id=507538
    .. [3] On estimating probabilities in tree pruning, equation 1,
        from https://link.springer.com/chapter/10.1007/BFb0017010
    .. [4] Additive smoothing, from https://en.wikipedia.org/wiki/Additive_smoothing#Generalized_to_the_case_of_known_incidence_rates
    .. [5] Target encoding done the right way https://maxhalford.github.io/blog/target-encoding/
    """

    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(
        self,
        verbose: int = 0,
        cols: list[str] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_missing: str = 'value',
        handle_unknown: str = 'value',
        quantile: float = 0.5,
        m: float = 1.0,
        min_group_size: int | float | None = None,
        min_group_name: str | None = None,
        combine_min_nan_groups: bool | str | None = None,
        composite_cols: list[tuple[str, ...]] | None = None,
        keep_components: bool = False,
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            min_group_size=min_group_size,
            min_group_name=min_group_name,
            combine_min_nan_groups=combine_min_nan_groups,
            composite_cols=composite_cols,
            keep_components=keep_components,
        )
        self.ordinal_encoder = None
        self.mapping = None
        self.quantile = quantile
        self.m = m

    def _fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None, **kwargs
    ) -> None:
        y = y.astype(float)
        if sample_weight is not None:
            sample_weight = _validate_sample_weight(sample_weight, len(y))

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value',
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        self.mapping = self.fit_quantile_encoding(X_ordinal, y, sample_weight=sample_weight)

    def fit_quantile_encoding(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> dict[str, pd.Series]:
        """Calculate the quantile encoding mapping.

        Parameters
        ----------
        X: training data.
        y: target data.
        sample_weight: optional frequency weights, one per row of ``y``. A row with
            weight ``w`` is equivalent to ``w`` copies of that row, without
            materializing them. Rows with zero weight are ignored. The per-category
            quantiles, the global prior quantile and the shrinkage counts are all
            computed from this same weighted data, so the result matches fitting on
            the expanded data.

        Returns
        -------
        mapping col-name -> series with category-label -> quantile mapping.
        """
        mapping = {}

        if sample_weight is None:
            weights = np.ones(len(y), dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        weighted = sample_weight is not None
        if weighted and not 0 <= self.quantile <= 1:
            raise ValueError(f'Quantiles must be in the range [0, 1], got {self.quantile}.')

        # Calculate global statistics. With weights this is the quantile of the
        # weighted sample, so zero-weight rows cannot move the prior.
        prior = (
            _weighted_quantile(y.to_numpy(dtype=float), weights, self.quantile)
            if weighted
            else np.quantile(y, self.quantile)
        )

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            if weighted:
                stats, n_positive_codes = _weighted_group_statistics(
                    X[col].to_numpy(), y.to_numpy(dtype=float), weights, self.quantile
                )
                # Categories with no positively weighted row (they only occur on
                # zero-weight rows) have no statistics of their own and fall back to
                # the prior, exactly like unseen categories in the expanded data.
                has_data = stats['count'] > 0
                estimate = pd.Series(prior, index=stats.index, dtype=float)
                counts = stats.loc[has_data, 'count']
                estimate.loc[has_data] = (
                    counts * stats.loc[has_data, 'quantile'] + prior * self.m
                ) / (counts + self.m)
                # Ignore unique columns. This helps to prevent overfitting on id-like
                # columns. Frequency weights repeat rows, so a column is only id-like
                # when every weighted row is its own category with weight 1.
                n_positive_rows = int((weights > 0).sum())
                is_id_like = (
                    n_positive_codes == n_positive_rows
                    and bool((weights[weights > 0] == 1).all())
                )
            else:
                # Calculate sum, count and quantile of the target for each unique value
                # in the feature col
                stats = y.groupby(X[col]).agg(
                    [lambda x: np.quantile(x, self.quantile), 'sum', 'count']
                )
                stats.columns = ['quantile', 'sum', 'count']

                # Calculate the m-probability estimate of the quantile
                estimate = (stats['count'] * stats['quantile'] + prior * self.m) / (
                    stats['count'] + self.m
                )
                is_id_like = len(stats) == len(y)

            # Ignore unique columns. This helps to prevent overfitting on id-like columns
            if is_id_like:
                estimate[:] = prior

            estimate = util.finalize_encoding_mapping(
                estimate, values, self.handle_unknown, self.handle_missing, prior
            )

            mapping[col] = estimate

        return mapping

    def _transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([util.UNKNOWN_SENTINEL]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        X = self.quantile_encode(X)
        return X

    def quantile_encode(self, X_in: pd.DataFrame) -> pd.DataFrame:
        """Apply quantile encoding."""
        X = X_in.copy(deep=True)

        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

        return X


# todo does not fit in schema since it is an ensemble of other encoders
class SummaryEncoder(BaseEstimator):
    """Summary Encoding for categorical features.

    It's an encoder designed for creating richer representations by applying quantile
    encoding for a set of quantiles.

    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    quantiles: list
        list of floats indicating the statistical quantiles. Each element represent a column
    m: float
        this is the “m” in the m-probability estimate. Higher value of m results into stronger
        shrinking. M is non-negative. 0 for no smoothing.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform
        (otherwise it will be a numpy array).
    handle_missing: str, int, float or callable
        options are 'error', 'return_nan'  and 'value', defaults to 'value',
        which returns the target quantile. A number is used as the encoded value
        for missing values that were not seen at fit time, and a callable
        fn(value, mapping) is evaluated once per column when the mapping is
        finalized during fit.
    handle_unknown: str, int, float or callable
        options are 'error', 'return_nan' and 'value', defaults to 'value',
        which returns the target quantile. A number is used as the encoded value
        for unseen categories, and a callable fn(value, mapping) is evaluated
        once per column when the mapping is finalized during fit.

    Notes
    -----
    Each quantile is encoded by an underlying QuantileEncoder, so columns in
    which every category is unique (id-like columns) are encoded with the
    prior quantile at every quantile, which prevents overfitting on
    identifier-like columns.

    Frequency weights passed as ``sample_weight`` to ``fit`` (and
    ``fit_transform``) are forwarded to every underlying ``QuantileEncoder``;
    see :class:`QuantileEncoder` for the exact semantics.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name='house_prices', as_frame=True)
    >>> display_cols = [
    ...     'Id',
    ...     'MSSubClass',
    ...     'MSZoning',
    ...     'LotFrontage',
    ...     'YearBuilt',
    ...     'Heating',
    ...     'CentralAir',
    ... ]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = SummaryEncoder(cols=['CentralAir', 'Heating'], quantiles=[0.25, 0.5, 0.75]).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 11 columns):
     #   Column         Non-Null Count  Dtype
    ---  ------         --------------  -----
     0   Id             1460 non-null   float64
     1   MSSubClass     1460 non-null   float64
     2   MSZoning       1460 non-null   object
     3   LotFrontage    1201 non-null   float64
     4   YearBuilt      1460 non-null   float64
     5   Heating_25     1460 non-null   float64
     6   Heating_50     1460 non-null   float64
     7   Heating_75     1460 non-null   float64
     8   CentralAir_25  1460 non-null   float64
     9   CentralAir_50  1460 non-null   float64
     10  CentralAir_75  1460 non-null   float64
    dtypes: float64(10), object(1)
    memory usage: 125.6+ KB
    None

    References
    ----------
    .. [1] Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems,
    https://link.springer.com/chapter/10.1007%2F978-3-030-85529-1_14
    .. [2] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification
    and Prediction Problems, equation 7, from https://dl.acm.org/citation.cfm?id=507538
    .. [3] On estimating probabilities in tree pruning, equation 1,
    from https://link.springer.com/chapter/10.1007/BFb0017010
    .. [4] Additive smoothing, from https://en.wikipedia.org/wiki/Additive_smoothing#Generalized_to_the_case_of_known_incidence_rates
    .. [5] Target encoding done the right way https://maxhalford.github.io/blog/target-encoding/
    """

    encoding_relation = util.EncodingRelation.ONE_TO_M

    def __init__(
        self,
        verbose: int = 0,
        cols: list[str] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_missing: str = 'value',
        handle_unknown: str = 'value',
        quantiles: Sequence[float] = (0.25, 0.75),
        m: float = 1.0,
    ):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.use_default_cols = (
            cols is None
        )  # if True, even a repeated call of fit() will select string columns from X
        self.use_all_cols = isinstance(cols, str) and cols.lower() == 'all'
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.quantiles = quantiles
        self.m = m
        self.encoder_list = None

    def fit(self, X: util.X_type, y: util.y_type, **kwargs) -> SummaryEncoder:
        """Fits the encoder according to X and y by fitting the individual encoders.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        kwargs : dict
            additional fit parameters forwarded to every underlying
            :class:`QuantileEncoder`, e.g. ``sample_weight``.

        Returns
        -------
        self : encoder
            Returns self.

        """
        X, y = util.convert_inputs(X, y)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        if self.use_all_cols:
            self.cols = X.columns.tolist()
        elif self.use_default_cols:
            self.cols = util.get_categorical_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        rounded_percentiles = [round(quantile * 100) for quantile in self.quantiles]
        if len(rounded_percentiles) != len(set(rounded_percentiles)):
            raise ValueError('There are two quantiles that belong to the same rounded percentile')

        encoder_list = []
        for quantile in self.quantiles:
            enc = QuantileEncoder(
                verbose=self.verbose,
                cols=self.cols,
                drop_invariant=self.drop_invariant,
                # always return df for individual encoders. If not desired this is handled below.
                return_df=True,
                handle_missing=self.handle_missing,
                handle_unknown=self.handle_unknown,
                quantile=quantile,
                m=self.m,
            )
            enc.fit(X.copy(), y, **kwargs)
            encoder_list.append(enc)
            self.drop_cols += enc.invariant_cols
        self.feature_names_out_ = reduce(
            operator.add,
            [
                (
                    [
                        self._get_col_name(c, enc.quantile)
                        for enc in encoder_list
                        if c not in enc.invariant_cols
                    ]
                    if c in self.cols
                    else [c]
                )
                for c in X.columns
            ],
        )
        self.encoder_list = encoder_list
        return self

    def transform(
        self, X: util.X_type, y: util.y_type | None = None, override_return_df: bool = False
    ) -> pd.DataFrame | np.ndarray:
        """Summary encode new data.

        Parameters
        ----------
        X: data to encode.
        y: optional target information.
        override_return_df: if true return a numpy array instead of a
            dataframe regardless of the return_df parameter.

        Returns
        -------
        encoded data.

        """
        if self.encoder_list is None:
            raise ValueError('Must train encoder before it can be used to transform data.')
        X, y = util.convert_inputs(X, y)

        orig_cols = X.columns
        transformed_df = X.copy()
        for idx, encoder in enumerate(self.encoder_list):
            colname_mapping = {col: self._get_col_name(col, encoder.quantile) for col in self.cols}
            X_encoded = encoder.transform(X.copy()).rename(columns=colname_mapping)
            if idx == 0:
                transformed_df = X_encoded
            else:
                new_feat = X_encoded[[c for c in X_encoded.columns if c not in orig_cols]]
                transformed_df = pd.concat([transformed_df, new_feat], axis=1)
        feature_order = [c for c in self.get_feature_names_out() if c in transformed_df]
        transformed_df = transformed_df[feature_order]

        if self.return_df or override_return_df:
            return transformed_df
        else:
            return transformed_df.to_numpy()

    def __sklearn_tags__(self) -> util.EncoderTags:
        """Set scikit transformer tags."""
        sk_tags = super().__sklearn_tags__()
        tags = util.EncoderTags.from_sk_tags(sk_tags)
        tags.target_tags.required = True
        return tags

    def fit_transform(self, X: util.X_type, y: util.y_type | None = None, **fit_params):
        """Fit and transform using target.

        This also uses the target for transforming, not only for training.
        """
        if y is None:
            raise TypeError('fit_transform() missing argument: ' 'y' '')
        return self.fit(X, y, **fit_params).transform(X, y)

    def get_feature_names(self) -> np.ndarray:
        """Deprecated method to get feature names. Use `get_feature_names_out` instead."""
        msg = (
            '`get_feature_names` is deprecated in all of sklearn. '
            'Use `get_feature_names_out` instead.'
        )
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        return self.get_feature_names_out()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Returns the names of all transformed / added columns.

        Note that in sklearn the get_feature_names_out function takes the feature_names_in
        as an argument and determines the output feature names using the input.
        A fit is usually not necessary and if so a NotFittedError is raised.
        We just require a fit all the time and return the fitted output columns.

        Returns
        -------
        feature_names: np.ndarray
            A list with all feature names transformed or added.
            Note: potentially dropped features (because the feature is constant/invariant)
            are not included!

        """
        out_feats = getattr(self, 'feature_names_out_', None)
        if not isinstance(out_feats, list):
            raise NotFittedError('Estimator has to be fitted to return feature names.')
        else:
            return np.array(out_feats, dtype=object)

    def get_feature_names_in(self) -> np.ndarray:
        """Get the names of all input columns present when fitting.

        These columns are necessary for the transform step.
        """
        in_feats = getattr(self, 'feature_names_in_', None)
        if isinstance(in_feats, list):
            in_feats = np.array(in_feats)
        if not isinstance(in_feats, np.ndarray):
            raise NotFittedError('Estimator has to be fitted to return feature names.')
        else:
            return in_feats

    @staticmethod
    def _get_col_name(col: str, quantile: float) -> str:
        percentile = round(quantile * 100)
        return f'{col}_{percentile}'
