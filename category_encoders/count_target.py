"""Count-based target encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder

__author__ = 'Obvious'


class CountTargetEncoder(util.SupervisedTransformerMixin, util.BaseEncoder):
    """Count-based target encoding with smoothing-adjusted log-odds.

    Supported targets: binary and multiclass classification. A continuous
    target raises NotImplementedError; regression via target binning is a
    planned follow-up (see issue #420).

    For every category of every encoded column, fit stores the per-class
    observation counts, the category size, and smoothing-adjusted log-odds
    against the global target prior. Transform emits those log-odds:

    - binary target: a single output column per encoded feature, matching the
      WOEEncoder column convention
    - multiclass target: one output column per class per encoded feature,
      named ``<column>_<class>``

    For a category c and class k let n_k(c) be the number of training rows in
    category c with class k, n(c) their sum, and prior_k the global class
    probability. Both are weighted sums when the fit receives sample weights
    (see below). The empirical class shares are blended with the prior by an
    S-shaped weight::

        w(c) = expit((n(c) - min_samples_leaf) / smoothing)
        p_smooth(k | c) = w(c) * n_k(c) / n(c) + (1 - w(c)) * prior_k

    and the encoded value is the log-evidence of the smoothed probability
    against the prior::

        binary:     log(p_smooth(1 | c) / p_smooth(0 | c)) - log(prior_1 / prior_0)
        multiclass: log(p_smooth(k | c) / prior_k)            (one column per class)

    Small categories are shrunk toward zero evidence, which tames the
    overfitting that raw counts would otherwise introduce on id-like columns.
    A category never observed at fit time encodes to 0 ("no evidence against
    the prior") under the default ``handle_unknown='value'``.

    Sample weights are supported through ``fit(X, y, sample_weight=...)``. The
    weight of a row is its multiplicity: counts become weighted counts (so a
    positive row of weight 3 counts as 3), and the smoothing support ``n(c)``
    and the global prior are weighted sums too. Integer weights therefore give
    exactly the same encoding as fitting on the correspondingly expanded data.
    A row with weight 0 does not take part in the fit at all: it neither
    creates a category nor contributes a class column or prior mass, and its
    categories are treated as unknown at transform time.

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
    handle_missing: str
        options are 'error', 'return_nan' and 'value', defaults to 'value',
        which treats missing values as a countable category at fit time.
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value',
        which maps unseen categories to zero evidence against the prior.
    min_samples_leaf: int
        category size at which the S-curve weight reaches 0.5. Categories
        smaller than this are dominated by the prior, larger ones by their
        own counts (parameter k in the original target-encoding paper).
    smoothing: float
        slope of the S-curve between category size and the prior/count blend.
        Higher values mean stronger regularization. The value must be strictly
        bigger than 0.

    Attributes
    ----------
    counts_ : dict
        Maps every encoded column to a DataFrame with the per-class counts
        observed at fit time (rows are the categories, columns the classes).
        With sample weights these are weighted counts, so the table can be
        checked against the fit input.

    Examples
    --------
    >>> from category_encoders import CountTargetEncoder
    >>> import pandas as pd
    >>> X = pd.DataFrame({'city': ['chicago', 'chicago', 'denver', 'denver', 'denver']})
    >>> y = [1, 0, 1, 1, 0]
    >>> enc = CountTargetEncoder().fit(X, y)
    >>> enc.transform(X)
           city
    0 -0.058774
    1 -0.058774
    2  0.043099
    3  0.043099
    4  0.043099

    Weighted fit: a positive row of weight 3 and a negative row of weight 1
    yield per-class counts of 3 and 1.

    >>> weights = [3, 1]
    >>> enc = CountTargetEncoder().fit(
    ...     pd.DataFrame({'city': ['chicago', 'chicago']}), [1, 0], sample_weight=weights
    ... )
    >>> enc.counts_['city']
             0  1
    chicago  1  3

    References
    ----------

    .. [1] Big Learning Made Easy with Counts (the "Dracula" count-based
        target scheme), from
        https://learn.microsoft.com/en-us/archive/blogs/machinelearning/big-learning-made-easy-with-counts

    """

    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_M

    _CONTINUOUS_TARGET_MSG = (
        'CountTargetEncoder supports classification targets only (binary or '
        'multiclass). A continuous target would first have to be discretized '
        'into bins, and encoding per bin is planned as a binning follow-up '
        '(see issue #420). For continuous targets please use TargetEncoder '
        'or GLMMEncoder.'
    )

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
    ) -> None:
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.ordinal_encoder = None
        self.mapping = None
        self.counts_ = None
        self._classes = None
        self._class_labels = None
        self._prior = None

    def _validate_sample_weight(
        self, sample_weight: util.y_type | None, n_rows: int
    ) -> np.ndarray | None:
        """Validate the fit-time sample weights and return them as a float array.

        Weights are multiplicities: a row of weight 3 behaves like three copies of
        that row. Zero weights are allowed (the row takes no part in the fit),
        negative or non-finite weights are not.
        """
        if sample_weight is None:
            return None
        weights = np.asarray(sample_weight, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != n_rows:
            raise ValueError(
                'sample_weight must be one-dimensional with one weight per row, got shape '
                f'{weights.shape} for {n_rows} rows.'
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError('sample_weight must not contain NaN or infinite values.')
        if (weights < 0).any():
            raise ValueError('sample_weight must not contain negative values.')
        return weights

    def _fit(self, X: util.X_type, y: util.y_type, sample_weight=None, **kwargs) -> None:
        # y arrives here label-encoded: bool/int stay, strings/categoricals are
        # coded by BaseEncoder.fit through LabelEncoder. Anything else (float,
        # complex) is a continuous target and out of scope for v1.
        if not (pd.api.types.is_bool_dtype(y) or pd.api.types.is_integer_dtype(y)):
            raise NotImplementedError(self._CONTINUOUS_TARGET_MSG)
        if self.smoothing <= 0:
            raise ValueError('smoothing must be strictly bigger than 0.')

        weights = self._validate_sample_weight(sample_weight, X.shape[0])
        if weights is None:
            X_fit, y_fit = X, y
        else:
            # zero-weight rows take no part in the fit: they must not register a
            # category, a class column or prior mass, so drop them before the
            # ordinal encoding even sees them
            keep = np.flatnonzero(weights > 0)
            if keep.size == 0:
                raise ValueError('All sample weights are zero, so there is nothing to fit on.')
            X_fit = X.iloc[keep]
            y_fit = y.iloc[keep]
            weights = weights[keep]

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose, cols=self.cols, handle_unknown='value', handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X_fit)
        X_ordinal = self.ordinal_encoder.transform(X_fit)

        self._classes = np.sort(y_fit.unique())
        if weights is None:
            class_totals = y_fit.value_counts().reindex(self._classes).to_numpy(dtype=float)
        else:
            # sum the row weights per class; self._classes is sorted so searchsorted
            # maps each label to its class position
            class_totals = np.bincount(
                np.searchsorted(self._classes, y_fit.to_numpy()),
                weights=weights,
                minlength=len(self._classes),
            )
        self._prior = pd.Series(class_totals / class_totals.sum(), index=self._classes)
        if self.lab_encoder_ is not None:
            self._class_labels = self.lab_encoder_.inverse_transform(self._classes)
        else:
            self._class_labels = self._classes

        # integer-weighted fits reproduce an expanded unweighted fit, including
        # its integral count table; fractional weights must stay as they are
        integral = weights is None or bool(np.all(weights == np.floor(weights)))
        self.mapping, self.counts_ = self._fit_mapping(X_ordinal, y_fit, weights, integral)

    def _fit_mapping(
        self,
        X_ordinal: pd.DataFrame,
        y: pd.Series,
        weights: np.ndarray | None = None,
        integral_counts: bool = True,
    ) -> tuple[dict[str, pd.Series | pd.DataFrame], dict[str, pd.DataFrame]]:
        """Compute per-column count tables and smoothed log-odds tables.

        Returns the transform-time mapping (ordinal code -> encoded value(s))
        and the inspectable per-class count tables (category -> counts). With
        sample weights the counts are weighted counts, so an integer weight acts
        as the row's multiplicity; the smoothing support and the prior are
        weighted sums as well.
        """
        mapping: dict[str, pd.Series | pd.DataFrame] = {}
        counts_tables: dict[str, pd.DataFrame] = {}
        is_binary = len(self._classes) == 2
        prior = self._prior.to_numpy(dtype=float)
        counts_dtype = np.int64 if integral_counts else float

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            # per-class counts per ordinal code; rows are observed codes only.
            # raw arrays, not Series: crosstab would align their (possibly
            # duplicated) indexes instead of pairing values positionally
            codes = X_ordinal[col].to_numpy()
            if weights is None:
                counts = pd.crosstab(codes, y.to_numpy())
            else:
                # weighted crosstab: absent (code, class) pairs are NaN rather
                # than 0 under aggfunc='sum', and reindex does not fill them
                counts = pd.crosstab(codes, y.to_numpy(), values=weights, aggfunc='sum')
                counts = counts.fillna(0.0)
            counts = counts.reindex(columns=self._classes, fill_value=0)
            counts_np = counts.to_numpy(dtype=float)
            n = counts_np.sum(axis=1)
            weight = expit((n - self.min_samples_leaf) / self.smoothing)
            shares = counts_np / n[:, None]
            smoothed = prior[None, :] * (1 - weight)[:, None] + shares * weight[:, None]

            if is_binary:
                table = pd.Series(
                    np.log(smoothed[:, 1] / smoothed[:, 0]) - np.log(prior[1] / prior[0]),
                    index=counts.index,
                    dtype=float,
                )
            else:
                class_cols = [f'{col}_{label}' for label in self._class_labels]
                table = pd.DataFrame(
                    np.log(smoothed / prior[None, :]), index=counts.index, columns=class_cols
                )

            self._add_sentinel_fills(table, values)

            mapping[col] = table
            # reindex the counts from codes back to the original categories
            code_to_category = pd.Series(values.index.to_numpy(), index=values.to_numpy())
            categories = code_to_category.reindex(counts.index).to_numpy()
            counts_tables[col] = pd.DataFrame(
                counts_np.astype(counts_dtype), index=categories, columns=self._class_labels
            )

        return mapping, counts_tables

    def _add_sentinel_fills(
        self, table: pd.Series | pd.DataFrame, ordinal_values: pd.Series
    ) -> None:
        """Fill the unknown (-1) and missing (-2) sentinel rows of a table."""
        if self.handle_unknown == 'return_nan':
            table.loc[-1] = np.nan
        elif self.handle_unknown == 'value':
            table.loc[-1] = 0.0

        if self.handle_missing == 'return_nan':
            # a NaN seen at fit has a real ordinal code; on clean fit data the
            # ordinal mapping pre-registers NaN -> -2 (see OrdinalEncoder)
            table.loc[ordinal_values.loc[np.nan]] = np.nan
        elif self.handle_missing == 'value':
            table.loc[-2] = 0.0

    def _transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        for col in self.cols:
            table = self.mapping[col]
            if isinstance(table, pd.Series):
                # binary: single log-odds column replaces the input in place
                X[col] = X[col].map(table)
            else:
                # multiclass: one column per class, inserted at the position
                # of the original column to preserve the column order
                positions = table.index.get_indexer(X[col].to_numpy())
                if (positions < 0).any():
                    raise ValueError(f'Unexpected category code found in column {col}')
                expanded = table.to_numpy()[positions]
                position = X.columns.get_loc(col)
                X = X.drop(columns=col)
                for class_idx, class_col in enumerate(table.columns):
                    X.insert(position + class_idx, class_col, expanded[:, class_idx])

        return X
