"""Target Encoder."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder

__author__ = 'chappers'


class TargetEncoder( util.SupervisedTransformerMixin,util.BaseEncoder):
    """Target encoding for categorical features.

    Supported targets: binomial and continuous.
    For polynomial target support, see PolynomialWrapper.

    For the case of categorical target: features are replaced with a blend of posterior
    probability of the target given particular categorical value and the prior probability
    of the target over all the training data.

    For the case of continuous target: features are replaced with a blend of the expected value
    of the target given particular categorical value and the expected value of the
    target over all the training data.

    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform
        (otherwise it will be a numpy array).
    handle_missing: str, int, float or callable
        options are 'error', 'return_nan'  and 'value', defaults to 'value',
        which returns the target mean. A number is used as the encoded value
        for missing values that were not seen at fit time, and a callable
        fn(value, mapping) is evaluated once per column when the mapping is
        finalized during fit.
    handle_unknown: str, int, float or callable
        options are 'error', 'return_nan' and 'value', defaults to 'value',
        which returns the target mean. A number is used as the encoded value
        for unseen categories, and a callable fn(value, mapping) is evaluated
        once per column when the mapping is finalized during fit.
    min_samples_leaf: int
        For regularization the weighted average between category mean and global mean is taken.
        The weight is an S-shaped curve between 0 and 1 with the number of samples for a category
        on the x-axis. The curve reaches 0.5 at min_samples_leaf.
        (parameter k in the original paper)
    smoothing: float
        smoothing effect to balance categorical average vs prior. Higher value means stronger
        regularization. The value must be strictly bigger than 0.
        Higher values mean a flatter S-curve (see min_samples_leaf).
    hierarchy: dict or dataframe
        A dictionary or a dataframe to define the hierarchy for mapping.

        If a dictionary, this contains a dict of columns to map into hierarchies.
        Dictionary key(s) should be the column name from X which requires mapping.
        For multiple hierarchical maps, this should be a dictionary of dictionaries.

        If dataframe: a dataframe defining columns to be used for the hierarchies.
        Column names must take the form:
            HIER_colA_1, ... HIER_colA_N, HIER_colB_1, ... HIER_colB_M, ...
        where [colA, colB, ...] are given columns in cols list.
        1:N and 1:M define the hierarchy for each column where 1 is the highest hierarchy
        (top of the tree).  A single column or multiple can be used, as relevant.

    Examples
    --------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> display_cols = [
    ...     'Id',
    ...     'MSSubClass',
    ...     'MSZoning',
    ...     'LotFrontage',
    ...     'YearBuilt',
    ...     'Heating',
    ...     'CentralAir',
    ... ]
    >>> bunch = fetch_openml(name='house_prices', as_frame=True)
    >>> y = bunch.target > 200000
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = TargetEncoder(cols=['CentralAir', 'Heating'], min_samples_leaf=20, smoothing=10).fit(
    ...     X, y
    ... )
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

    >>> from category_encoders.datasets import load_compass
    >>> X, y = load_compass()
    >>> hierarchical_map = {'compass': {'N': ('N', 'NE'), 'S': ('S', 'SE'), 'W': 'W'}}
    >>> enc = TargetEncoder(
    ...     verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map, cols=['compass']
    ... ).fit(X.loc[:, ['compass']], y)
    >>> hierarchy_dataset = enc.transform(X.loc[:, ['compass']])
    >>> print(hierarchy_dataset['compass'].values)
    [0.62263617 0.62263617 0.90382995 0.90382995 0.90382995 0.17660024
     0.17660024 0.46051953 0.46051953 0.46051953 0.46051953 0.40332791
     0.40332791 0.40332791 0.40332791 0.40332791]
    >>> X, y = load_postcodes('binary')
    >>> cols = ['postcode']
    >>> HIER_cols = ['HIER_postcode_1', 'HIER_postcode_2', 'HIER_postcode_3', 'HIER_postcode_4']
    >>> enc = TargetEncoder(
    ...     verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=X[HIER_cols], cols=['postcode']
    ... ).fit(X['postcode'], y)
    >>> hierarchy_dataset = enc.transform(X['postcode'])
    >>> print(hierarchy_dataset.loc[0:10, 'postcode'].values)
    [0.75063473 0.90208756 0.88328833 0.77041254 0.68891504 0.85012847
    0.76772574 0.88742357 0.7933824  0.63776756 0.9019973 ]

    References
    ----------

    .. [1] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification
    and Prediction Problems, from https://dl.acm.org/citation.cfm?id=507538

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
        min_samples_leaf: int = 20,
        smoothing: float = 10,
        hierarchy: dict = None,
        min_group_size: int | float | None = None,
        min_group_name: str | None = None,
        combine_min_nan_groups: bool | str | None = None,
        composite_cols: list[tuple[str, ...]] | None = None,
        keep_components: bool = False,
    ) -> None:
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
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.hierarchy = hierarchy
        self.ordinal_encoder = None
        self.mapping = None
        self._mean = None
        # Fitted hierarchy state used to back unseen leaves at transform time
        # with the deepest available ancestor's smoothed encoding.
        self._hier_ordinal_encoder = None
        self._hier_level_mappings = None
        # Row-aligned hierarchy supplied to a single transform() call, if any.
        self._inference_hierarchy = None
        # Boolean mask of unknown leaves that no fitted ancestor could resolve.
        self._unresolved_unknown = None
        # Call this in the constructor only for the possible side effect of raising an exception.
        self._generate_inverted_hierarchy()

    @staticmethod
    def _check_dict_key_tuples(dict_to_check: dict[Any, tuple]) -> tuple[bool, int]:
        """Check if all tuples in the dict values have the same length.

        Parameters
        ----------
        dict_to_check: dictionary to check

        Returns
        -------
        tuple: first entry if all sizes are equal, second minimum size.
        """
        min_tuple_size = min(len(v) for v in dict_to_check.values())
        max_tuple_size = max(len(v) for v in dict_to_check.values())
        return min_tuple_size == max_tuple_size, min_tuple_size

    def _fit(self, X: util.X_type, y: util.y_type, **kwargs) -> None:
        inverted_hierarchy, self.hierarchy_depth = self._generate_inverted_hierarchy()
        if isinstance(inverted_hierarchy, dict):
            X_hier = pd.DataFrame()
            for switch in inverted_hierarchy:
                if switch in self.cols:
                    colnames = [
                        f'HIER_{str(switch)}_{str(i + 1)}'
                        for i in range(self.hierarchy_depth[switch])
                    ]
                    df = pd.DataFrame(
                        X[str(switch)].map(inverted_hierarchy[str(switch)]).tolist(),
                        index=X.index,
                        columns=colnames,
                    )
                    X_hier = pd.concat([X_hier, df], axis=1)
        elif isinstance(inverted_hierarchy, pd.DataFrame):
            X_hier = inverted_hierarchy

        if isinstance(inverted_hierarchy, (dict, pd.DataFrame)):
            self._hier_ordinal_encoder = OrdinalEncoder(
                verbose=self.verbose,
                cols=X_hier.columns,
                handle_unknown='value',
                handle_missing='value',
            )
            self._hier_ordinal_encoder = self._hier_ordinal_encoder.fit(X_hier)
            X_hier_ordinal = self._hier_ordinal_encoder.transform(X_hier)

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose, cols=self.cols, handle_unknown='value', handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        self._hier_level_mappings = {}
        if inverted_hierarchy is not None:
            self.mapping = self.fit_target_encoding(
                pd.concat([X_ordinal, X_hier_ordinal], axis=1), y
            )
        else:
            self.mapping = self.fit_target_encoding(X_ordinal, y)

    def fit_target_encoding(self, X: util.X_type, y: util.y_type) -> dict[str, np.ndarray]:
        """Fit the target encoding mapping.

        Parameters
        ----------
        X: training data to fit on.
        y: training target.

        Returns
        -------
        dictionary: column -> encoding values for column

        """
        mapping = {}
        prior = self._mean = y.mean()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            if 'HIER_' not in str(col):
                values = switch.get('mapping')

                scalar = prior
                if (isinstance(self.hierarchy, dict) and col in self.hierarchy) or (
                    isinstance(self.hierarchy, pd.DataFrame)
                ):
                    self._hier_level_mappings[col] = {}
                    for i in range(self.hierarchy_depth[col]):
                        col_hier = 'HIER_' + str(col) + '_' + str(i + 1)
                        col_hier_m1 = (
                            col
                            if i == self.hierarchy_depth[col] - 1
                            else 'HIER_' + str(col) + '_' + str(i + 2)
                        )
                        if not X[col].equals(X[col_hier]) and len(X[col_hier].unique()) > 1:
                            stats_hier = y.groupby(X[col_hier]).agg(['count', 'mean'])
                            smoove_hier = self._weighting(stats_hier['count'])
                            scalar_hier = (
                                scalar * (1 - smoove_hier) + stats_hier['mean'] * smoove_hier
                            )
                            # Keep the smoothed value of every node at this level so that
                            # unseen leaves can fall back to their deepest fitted ancestor
                            # at transform time. The series is indexed by the ordinal codes
                            # of ``col_hier`` (the parent level of ``col_hier_m1``).
                            self._hier_level_mappings[col][i] = scalar_hier
                            scalar_hier_long = X[[col_hier_m1, col_hier]].drop_duplicates()
                            scalar_hier_long.index = np.arange(1, scalar_hier_long.shape[0] + 1)
                            scalar = scalar_hier_long[col_hier].map(scalar_hier.to_dict())

                stats = y.groupby(X[col]).agg(['count', 'mean'])
                smoove = self._weighting(stats['count'])

                smoothing = scalar * (1 - smoove) + stats['mean'] * smoove

                smoothing = util.finalize_encoding_mapping(
                    smoothing, values, self.handle_unknown, self.handle_missing, prior
                )

                mapping[col] = smoothing

        return mapping

    def transform(
        self,
        X: util.X_type,
        y: util.y_type | None = None,
        override_return_df: bool = False,
        hierarchy: dict | pd.DataFrame | None = None,
    ) -> pd.DataFrame | np.ndarray:
        """Transform categorical features using the fitted target encoding.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Data to transform.
        y: array-like, shape = [n_samples] or None
            Optional target, retained for API compatibility.
        override_return_df: bool
            Override ``return_df`` to force a DataFrame result.
        hierarchy: dict or dataframe, optional
            Hierarchy information for the rows being transformed, used to back
            unseen leaves with the smoothed value of their deepest fitted
            ancestor.

            As a dictionary, it maps an encoded column name to a dict with the
            same nested form as the fit-time ``hierarchy`` argument, e.g.
            ``{'store': {'North': ('unknown', ...), 'South': ('other', ...)}}``.

            As a dataframe, it is row-aligned with ``X`` and uses the fitted
            ``HIER_<col>_<level>`` column names, containing the ancestor label
            at every level for each row.

            When omitted, unseen categories keep the ordinary
            ``handle_unknown`` fallback (the global target mean by default).
            Passing a hierarchy to an encoder that was fitted without one
            raises a ``ValueError``.

        Returns
        -------
        transformed: dataframe or ndarray
        """
        self._inference_hierarchy = self._prepare_inference_hierarchy(X, hierarchy)
        try:
            return super().transform(X, y, override_return_df=override_return_df)
        finally:
            self._inference_hierarchy = None

    def _prepare_inference_hierarchy(
        self, X: util.X_type, hierarchy: dict | pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """Turn the transform-time hierarchy input into fitted HIER_ columns."""
        if hierarchy is None:
            return None
        if self._hier_ordinal_encoder is None or not self._hier_level_mappings:
            raise ValueError(
                'A hierarchy was provided at transform time, but this encoder '
                'was fitted without a hierarchy.'
            )

        fitted_hier_cols = self._hier_ordinal_encoder.cols
        if isinstance(X, pd.DataFrame):
            X_frame = X
        else:
            # Array-like input has no column names yet; reattach the fitted
            # feature names positionally, mirroring the base transform flow.
            X_frame = pd.DataFrame(X, columns=getattr(self, 'feature_names_in_', None))

        if isinstance(hierarchy, dict):
            parts = []
            for col in self.cols:
                if col not in hierarchy:
                    continue
                if col not in self._hier_level_mappings:
                    raise ValueError(
                        f'Transform-time hierarchy references column "{col}", '
                        'which has no fitted hierarchy.'
                    )
                inverted = util.flatten_reverse_dict(hierarchy[col])
                tuple_ok, depth = self._check_dict_key_tuples(inverted)
                if not tuple_ok or depth != self.hierarchy_depth[col]:
                    raise ValueError(
                        f'Transform-time hierarchy for column "{col}" must have '
                        f'depth {self.hierarchy_depth[col]} with equal level sizes.'
                    )
                leaf_to_path = {k: v for t, v in inverted.items() for k in t}
                colnames = [
                    f'HIER_{col}_{i + 1}' for i in range(self.hierarchy_depth[col])
                ]
                parts.append(
                    pd.DataFrame(
                        X_frame[col].map(leaf_to_path).tolist(),
                        index=X_frame.index,
                        columns=colnames,
                    )
                )
            if not parts:
                raise ValueError(
                    'The transform-time hierarchy does not cover any encoded column.'
                )
            inference_hier = pd.concat(parts, axis=1)
        elif isinstance(hierarchy, pd.DataFrame):
            missing = [c for c in fitted_hier_cols if c not in hierarchy.columns]
            if missing:
                raise ValueError(
                    'Transform-time hierarchy dataframe is missing columns: '
                    f'{missing}'
                )
            inference_hier = hierarchy[fitted_hier_cols].copy()
        else:
            raise ValueError('Given hierarchy mapping is neither a dictionary nor a dataframe')

        if len(inference_hier) != len(X_frame):
            raise ValueError(
                'The transform-time hierarchy must be row-aligned with X '
                f'({len(inference_hier)} hierarchy rows vs {len(X_frame)} data rows).'
            )
        inference_hier.index = X_frame.index
        # Reuse the fitted level encoders: unseen hierarchy nodes become the
        # unknown sentinel and therefore cannot act as a fallback.
        return self._hier_ordinal_encoder.transform(inference_hier)

    def _transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        # Now X is the correct dimensions it works with pre fitted ordinal encoder
        X = self.ordinal_encoder.transform(X)
        inference_hierarchy = getattr(self, '_inference_hierarchy', None)

        if self.handle_unknown == 'error' and inference_hierarchy is None:
            if X[self.cols].isin([util.UNKNOWN_SENTINEL]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        X = self.target_encode(X)

        if self.handle_unknown == 'error' and inference_hierarchy is not None:
            # Unknown leaves without a resolvable ancestor still violate 'error'.
            if self._unresolved_unknown.any().any():
                raise ValueError('Unexpected categories found in dataframe')
        return X

    def target_encode(self, X_in: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding via encoder mapping."""
        X = X_in.copy(deep=True)
        inference_hierarchy = getattr(self, '_inference_hierarchy', None)
        self._unresolved_unknown = pd.DataFrame(False, index=X.index, columns=self.cols)

        # Was not mapping extra columns as self.feature_names_in did not include new column
        for col in self.cols:
            encoded = X[col].map(self.mapping[col])
            if inference_hierarchy is not None and col in self._hier_level_mappings:
                encoded, unresolved = self._apply_hierarchy_fallback(
                    col, X[col], encoded, inference_hierarchy
                )
                self._unresolved_unknown[col] = unresolved
            X[col] = encoded

        return X

    def _apply_hierarchy_fallback(
        self,
        col: str,
        ordinal_codes: pd.Series,
        encoded: pd.Series,
        inference_hierarchy: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        """Back unseen leaves with their deepest fitted ancestor value.

        Returns the encoded series and a boolean mask of unknown leaves whose
        hierarchy nodes were all unobserved at fit time.
        """
        unknown_rows = ordinal_codes == util.UNKNOWN_SENTINEL
        if not unknown_rows.any():
            return encoded, pd.Series(False, index=ordinal_codes.index)

        fallback = pd.Series(np.nan, index=ordinal_codes.index, dtype=float)
        # Walk from the deepest fitted level towards the root; the first level
        # whose node was observed at fit time wins.
        for level in sorted(self._hier_level_mappings[col], reverse=True):
            hier_col = f'HIER_{col}_{level + 1}'
            if hier_col not in inference_hierarchy.columns:
                continue
            unresolved = fallback.isna()
            if not unresolved.any():
                break
            level_mapping = self._hier_level_mappings[col][level]
            codes = inference_hierarchy[hier_col]
            known_codes = codes.isin(level_mapping.index)
            eligible = unresolved & unknown_rows & known_codes
            fallback[eligible] = codes[eligible].map(level_mapping)

        resolved = unknown_rows & fallback.notna()
        # Rows without a resolvable ancestor (or without hierarchy information)
        # keep the ordinary handle_unknown encoding, e.g. the global target mean.
        return fallback.fillna(encoded), unknown_rows & ~resolved

    def _weighting(self, n: int) -> float:
        # monotonically increasing function of n bounded between 0 and 1
        # sigmoid in this case, using scipy.expit for numerical stability
        return expit((n - self.min_samples_leaf) / self.smoothing)

    def _generate_inverted_hierarchy(self) -> tuple[dict | pd.DataFrame | None, dict]:
        if isinstance(self.hierarchy, (dict, pd.DataFrame)) and self.cols is None:
            raise ValueError('Hierarchy is defined but no columns are named for encoding')
        if isinstance(self.hierarchy, dict):
            inverted_hierarchy = {}
            hierarchy_depth = {}
            for switch in self.hierarchy:
                flattened_hierarchy = util.flatten_reverse_dict(self.hierarchy[switch])
                hierarchy_check = self._check_dict_key_tuples(flattened_hierarchy)
                hierarchy_depth[switch] = hierarchy_check[1]
                if not hierarchy_check[0]:
                    raise ValueError(
                        'Hierarchy mapping contains different levels for key "' + switch + '"'
                    )
                inverted_hierarchy[switch] = {
                    (k if isinstance(t, tuple) else t): v
                    for t, v in flattened_hierarchy.items()
                    for k in t
                }
        elif isinstance(self.hierarchy, pd.DataFrame):
            inverted_hierarchy = self.hierarchy
            hierarchy_depth = {}
            for col in self.cols:
                HIER_cols = inverted_hierarchy.columns[
                    inverted_hierarchy.columns.str.startswith(f'HIER_{col}')
                ].tolist()
                HIER_levels = [int(i.replace(f'HIER_{col}_', '')) for i in HIER_cols]
                if np.array_equal(sorted(HIER_levels), np.arange(1, max(HIER_levels) + 1)):
                    hierarchy_depth[col] = max(HIER_levels)
                else:
                    raise ValueError(f'Hierarchy columns are not complete for column {col}')
        elif self.hierarchy is None:
            inverted_hierarchy = None
            hierarchy_depth = {}
        else:
            raise ValueError('Given hierarchy mapping is neither a dictionary nor a dataframe')
        return inverted_hierarchy, hierarchy_depth

