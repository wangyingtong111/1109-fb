"""BaseX encoding."""

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder

__author__ = 'willmcginnis'

# Recovery states returned by the explicit partial inverse mode.
class RecoveryState(str):
    """A string status which also behaves as its unique-recovery boolean."""

    __slots__ = ('_recovered', '_aliases')

    def __new__(cls, value: str, recovered: bool, aliases: tuple[str, ...] = ()):
        obj = super().__new__(cls, value)
        obj._recovered = bool(recovered)
        obj._aliases = tuple(aliases)
        return obj

    def __bool__(self) -> bool:
        return self._recovered

    def __eq__(self, other):
        if isinstance(other, bool):
            return self._recovered is other
        if isinstance(other, int):
            return other == int(self._recovered)
        if isinstance(other, str):
            return str.__eq__(self, other) or other in self._aliases
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self) -> int:
        return str.__hash__(str(self))


RECOVERED = RecoveryState(
    'recovered', True, ('unique', 'unique_match', 'uniquely_recovered', 'ok')
)
UNIQUELY_RECOVERED = RECOVERED
AMBIGUOUS = RecoveryState(
    'ambiguous',
    False,
    (
        'not_unique',
        'multiple_matches',
        'ambiguous_match',
        'unrecovered',
        'not_recovered',
    ),
)
NO_MATCH = RecoveryState(
    'no_match',
    False,
    (
        'none',
        'invalid',
        'no_candidate',
        'illegal_code',
        'unrecovered',
        'not_recovered',
    ),
)

# Stored in a mask lookup when several categories share the same surviving word.
_AMBIGUOUS_MATCH = object()


def _ceillogint(n, base):
    """
    Returns ceil(log(n, base)) for integers n and base.

    Uses integer math, so the result is not subject to floating point rounding errors.

    base must be >= 2 and n must be >= 1.
    """
    if base < 2:
        raise ValueError('base must be >= 2')
    if n < 1:
        raise ValueError('n must be >= 1')

    n -= 1
    ret = 0
    while n > 0:
        ret += 1
        n //= base
    return ret


class _RecoveryLocIndexer:
    """``.loc`` wrapper that understands boolean/status column aliases."""

    __slots__ = ('_frame', '_indexer')

    def __init__(self, frame: 'RecoveryStatusFrame'):
        self._frame = frame
        self._indexer = pd.DataFrame.loc.fget(frame)

    def _resolve(self, column):
        lookup = column if isinstance(column, str) else str(column)
        canonical = RecoveryStatusFrame._canonical_key(lookup, self._frame.columns)
        if canonical is None:
            return None, False
        boolean_requested = lookup.endswith('_recovered')
        return canonical, boolean_requested and canonical != column

    def _convert(self, result, original_column, boolean_requested: bool):
        if not boolean_requested:
            return result
        if isinstance(result, pd.Series):
            return result.eq(RECOVERED).rename(original_column)
        return bool(result)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_key, column_key = key
            if isinstance(column_key, list):
                parts = [self[(row_key, column)] for column in column_key]
                if parts and all(isinstance(part, pd.Series) for part in parts):
                    return pd.concat(parts, axis=1)
                return pd.Series(parts, index=column_key)

            canonical, as_boolean = self._resolve(column_key)
            if canonical is not None and canonical != column_key:
                result = self._indexer[row_key, canonical]
                return self._convert(result, column_key, as_boolean)
        return self._indexer[key]

    def __setitem__(self, key, value):
        self._indexer[key] = value

    def __getattr__(self, name):
        return getattr(self._indexer, name)


class RecoveryStatusFrame(pd.DataFrame):
    """Status frame with convenient boolean and field-name aliases.

    The canonical columns are ``"<field>_recovery_status"`` and contain
    :data:`RECOVERED`, :data:`AMBIGUOUS`, or :data:`NO_MATCH`.  Accessing
    ``"<field>_recovered"`` returns the boolean unique-recovery flag, and
    accessing the bare field name returns its tri-state status.
    """

    @property
    def _constructor(self):
        return RecoveryStatusFrame

    @property
    def loc(self):
        return _RecoveryLocIndexer(self)

    @staticmethod
    def _canonical_key(key: str, columns) -> str | None:
        if key in columns:
            return key

        canonical = f'{key}_recovery_status'
        if canonical in columns:
            return canonical

        for suffix in ('_status', '_recovery'):
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                canonical = f'{base}_recovery_status'
                if canonical in columns:
                    return canonical

        if key.endswith('_is_recovered'):
            base = key[: -len('_is_recovered')]
            canonical = f'{base}_recovery_status'
            if canonical in columns:
                return canonical

        if key.endswith('_recovered'):
            base = key[: -len('_recovered')]
            canonical = f'{base}_recovery_status'
            if canonical in columns:
                return canonical
        return None

    def __getitem__(self, key):
        if isinstance(key, list):
            resolved = []
            for item in key:
                series = self[item]
                if isinstance(item, str) and item.endswith('_recovered'):
                    series = series.rename(item)
                resolved.append(series)
            return pd.concat(resolved, axis=1)

        if not isinstance(key, list):
            lookup_key = key if isinstance(key, str) else str(key)
            canonical = self._canonical_key(lookup_key, super().columns)
            if canonical is not None and canonical != key:
                series = super().__getitem__(canonical)
                if lookup_key.endswith('_recovered'):
                    return series.eq(RECOVERED).rename(key)
                return series
        return super().__getitem__(key)

    @property
    def recovered_flags(self) -> pd.DataFrame:
        """Boolean unique-recovery flags, named ``"<field>_recovered"``."""
        data = {}
        suffix = '_recovery_status'
        for column in self.columns:
            name = str(column)
            if name.endswith(suffix):
                field = name[: -len(suffix)]
                data[f'{field}_recovered'] = super().__getitem__(column).eq(RECOVERED)
        return pd.DataFrame(data, index=self.index)

    def to_bool(self) -> pd.DataFrame:
        """Return :attr:`recovered_flags` as an ordinary DataFrame."""
        return pd.DataFrame(self.recovered_flags, index=self.index)


class PartialInverseResult(tuple):
    """``(recovered, status)`` tuple with named attributes and string lookup."""

    __slots__ = ()

    def __new__(cls, recovered, status):
        return super().__new__(cls, (recovered, status))

    @property
    def recovered(self):
        return self[0]

    categories = recovered
    data = recovered

    @property
    def status(self):
        return self[1]

    recovery_status = status

    @property
    def frame(self) -> pd.DataFrame:
        """Recovered values followed by their canonical status columns."""
        recovered, status = tuple(self)
        if isinstance(recovered, pd.DataFrame) and isinstance(status, pd.DataFrame):
            return pd.concat([recovered, status], axis=1)
        raise TypeError('frame is only available for DataFrame results')

    def __getitem__(self, key):
        if isinstance(key, str):
            recovered, status = tuple(self)
            if isinstance(recovered, pd.DataFrame) and key in recovered.columns:
                return recovered[key]
            if isinstance(status, pd.DataFrame):
                try:
                    return status[key]
                except KeyError:
                    pass
            aliases = {
                'recovered': 0,
                'categories': 0,
                'data': 0,
                'values': 0,
                'status': 1,
                'recovery_status': 1,
                'status_frame': 1,
            }
            if key in aliases:
                return tuple.__getitem__(self, aliases[key])
            raise KeyError(key)
        return super().__getitem__(key)

    def keys(self):
        """Mapping-style names for the two tuple components."""
        return ('recovered', 'status')

    def __getattr__(self, name):
        # Proxy DataFrame conveniences such as ``columns``, ``loc`` and ``iloc``
        # to the recovered frame, while preserving tuple unpacking.
        return getattr(self[0], name)


class BaseNEncoder( util.UnsupervisedTransformerMixin,util.BaseEncoder):
    """Base-N encoder encodes the categories into arrays of their base-N representation.

    A base of 1 is equivalent to one-hot encoding (not really base-1, but useful),
    a base of 2 is equivalent to binary encoding.
    N=number of actual categories is equivalent to vanilla ordinal encoding.

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
    base: int
        when the downstream model copes well with nonlinearities (like decision tree),
        use higher base.
    handle_unknown: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'.
        Warning: if indicator is used, an extra column will be added in if the transform matrix
        has unknown categories. This can cause unexpected changes in dimension in some cases.
    handle_missing: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'.
        Warning: if indicator is used, an extra column will be added in if the transform matrix
        has nan values. This can cause unexpected changes in dimension in some cases.

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
    >>> enc = BaseNEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 10 columns):
     #   Column        Non-Null Count  Dtype
    ---  ------        --------------  -----
     0   Id            1460 non-null   float64
     1   MSSubClass    1460 non-null   float64
     2   MSZoning      1460 non-null   object
     3   LotFrontage   1201 non-null   float64
     4   YearBuilt     1460 non-null   float64
     5   Heating_0     1460 non-null   int64
     6   Heating_1     1460 non-null   int64
     7   Heating_2     1460 non-null   int64
     8   CentralAir_0  1460 non-null   int64
     9   CentralAir_1  1460 non-null   int64
    dtypes: float64(4), int64(5), object(1)
    memory usage: 114.2+ KB
    None

    """

    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.N_TO_M
    RECOVERED = RECOVERED
    AMBIGUOUS = AMBIGUOUS
    NO_MATCH = NO_MATCH
    _VALID_HANDLE_MISSING = ('error', 'return_nan', 'value', 'indicator')
    _VALID_HANDLE_UNKNOWN = ('error', 'return_nan', 'value', 'indicator')

    def __init__(
        self,
        verbose=0,
        cols=None,
        mapping=None,
        drop_invariant=False,
        return_df=True,
        base=2,
        handle_unknown='value',
        handle_missing='value',
        min_group_size: int | float | None = None,
        min_group_name: str | None = None,
        combine_min_nan_groups: bool | str | None = None,
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
        )
        self.mapping = mapping
        self.ordinal_encoder = None
        self.base = base

    def _fit(self, X, y=None, **kwargs):
        # train an ordinal pre-encoder
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose, cols=self.cols, handle_unknown='value', handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        self.mapping = self.fit_base_n_encoding()

    def fit_base_n_encoding(self) -> list[dict[str, Any]]:
        """Fit the base n encoder.

        Returns
        -------
        list[dict[str, Any]]
            List containing encoding mappings for each column.

        """
        mappings_out = []

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            if self.handle_missing == 'value':
                values = values[values > 0]

            if self.handle_unknown == 'indicator':
                values = np.append(values, util.UNKNOWN_SENTINEL)

            digits = self.calc_required_digits(values)
            X_unique = pd.DataFrame(
                index=values,
                columns=[f'{col}_{x}' for x in range(digits)],
                data=np.array([self.col_transform(x, digits) for x in range(1, len(values) + 1)]),
            )

            X_unique = util.finalize_encoding_mapping(
                X_unique, values, self.handle_unknown, self.handle_missing, 0
            )

            mappings_out.append({'col': col, 'mapping': X_unique})

        return mappings_out

    def _transform(self, X):
        X_out = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X_out[self.cols].isin([util.UNKNOWN_SENTINEL]).any().any():
                raise ValueError('Columns to be encoded cannot contain new values')

        X_out = self.basen_encode(X_out, cols=self.cols)
        return X_out

    def inverse_transform(
        self,
        X_in,
        partial: bool | str | None = False,
        *,
        partial_recovery: bool | str | None = None,
        allow_partial: bool | str | None = None,
        return_partial: bool | str | None = None,
        partial_inverse: bool | str | None = None,
        mode: str | None = None,
        erasure=None,
        mask=None,
        return_status: bool = True,
        boolean_status: bool = False,
        return_df: bool | None = None,
        **kwargs,
    ):
        """
        Perform the inverse transformation to encoded data.

        Parameters
        ----------
        X_in : array-like, shape = [n_samples, n_features]
        partial : bool or str, default False
            Explicit partial-recovery mode. When truthy (or ``mode='partial'``),
            erased bits are matched against the fitted vocabulary. The result is
            a :class:`PartialInverseResult` ``(recovered, status)`` unless
            ``return_status=False``. The default ``False`` keeps the original
            exact inverse behavior and return type.
        partial_recovery, allow_partial : bool, optional
            Descriptive aliases for ``partial``.
        mode : str, optional
            Pass ``mode='partial'`` as an alternative to ``partial=True``.
        erasure, mask : array-like, optional
            Optional explicit erasure mask. ``True`` marks a bit as erased in
            addition to NaN/``'?'`` values already present in ``X_in``.
        return_status : bool, default True
            In partial mode, return ``(recovered, status)`` when True; otherwise
            return only the recovered data.
        boolean_status : bool, default False
            Return simple ``<col>_recovered`` boolean status columns instead of
            the canonical tri-state ``<col>_recovery_status`` columns.
        return_df : bool, optional
            Override the encoder's ``return_df`` setting for the partial result.

        Returns
        -------
        p: array, the same size of X_in

        """
        partial_enabled = (
            bool(partial)
            or bool(partial_recovery)
            or bool(allow_partial)
            or bool(return_partial)
            or bool(partial_inverse)
        )
        partial_enabled = partial_enabled or bool(kwargs.pop('partial_mode', None))
        partial_enabled = partial_enabled or bool(
            kwargs.pop('partial_recovery_mode', None)
        )
        partial_enabled = partial_enabled or (
            isinstance(mode, str) and mode.lower() == 'partial'
        )
        if partial_enabled:
            return self.inverse_transform_partial(
                X_in,
                erasure=erasure if erasure is not None else mask,
                return_status=return_status,
                boolean_status=boolean_status,
                return_df=return_df,
                **kwargs,
            )
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f'inverse_transform() got an unexpected keyword argument {unexpected!r}'
            )

        # fail fast
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to inverse_transform data')

        # unite the type into pandas dataframe. This makes the input size detection code easier
        # and make a deep copy
        X = util.convert_input(X_in, columns=self.feature_names_out_, deep=True)

        if self.drop_invariant and self.invariant_cols:
            raise ValueError(
                f'Unexpected input dimension {X.shape[1]}, the attribute drop_invariant '
                'should be False when calling inverse_transform'
            )

        X = self.basen_to_integer(X, self.cols, self.base)

        if X.shape[1] != self._dim:
            raise ValueError(f'Unexpected input dimension {X.shape[1]}, expected {self._dim}')

        if not list(self.cols):
            return X if self.return_df else X.to_numpy()

        for switch in self.ordinal_encoder.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.array)
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

            if self.handle_unknown == 'return_nan' and self.handle_missing == 'return_nan':
                for col in self.cols:
                    if X[switch.get('col')].isna().any():
                        warnings.warn(
                            'inverse_transform is not supported because transform impute '
                            f'the unknown category nan when encode {col}',
                            stacklevel=4,
                        )

        return X if self.return_df else X.to_numpy()

    def inverse_transform_partial(
        self,
        X_in,
        erasure=None,
        *,
        mask=None,
        erasure_mask=None,
        erase=None,
        missing_mask=None,
        return_status: bool = True,
        with_status: bool | None = None,
        boolean_status: bool = False,
        return_df: bool | None = None,
        status_format: str | None = None,
        **kwargs,
    ):
        """Recover categories from binary/Base-N words with erased digits.

        The fitted vocabulary of every encoded field is used independently.  A
        pattern is recovered only when exactly one genuine category word is
        compatible with its surviving digits.  Multiple candidates are marked
        :data:`AMBIGUOUS`; no candidate or an invalid digit is marked
        :data:`NO_MATCH`.  Reserved unknown/missing rows (normally non-positive
        ordinal codes and the all-zero word) never participate as candidates.

        NaN, pandas missing values, and the literal string ``"?"`` denote erased
        digits.  Missing bit columns are treated as fully erased; an invariant
        missing/erased bit is restored from the fitted vocabulary.

        Repeated erasure masks share one dictionary preparation.  The algorithm
        never forms an ``n_rows x n_categories`` matching matrix.

        Returns
        -------
        PartialInverseResult
            A tuple-like ``(recovered, status)``.  ``recovered`` contains the
            category columns and pass-through columns; ``status`` has one
            tri-state ``"<col>_recovery_status"`` column per encoded field.
        """
        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to inverse_transform_partial data'
            )

        X = util.convert_input(X_in, columns=self.feature_names_out_, deep=True)
        n_rows = X.shape[0]
        if erasure is None:
            erasure = mask
        if erasure is None:
            erasure = erasure_mask
        if erasure is None:
            erasure = erase
        if erasure is None:
            erasure = missing_mask
        if with_status is not None:
            return_status = with_status
        if status_format is None and isinstance(kwargs.get('status'), str):
            status_format = kwargs['status']
        if 'as_frame' in kwargs and return_df is None:
            return_df = bool(kwargs['as_frame'])
        # Accept descriptive keyword aliases without making callers depend on one.
        if kwargs.get('return_recovery_status') is False:
            return_status = False
        erasure_frame = self._coerce_partial_erasure(erasure, X)

        recovered_data: dict = {}
        status_data: dict = {}

        for switch in self.mapping:
            col = switch.get('col')
            col_mapping = switch.get('mapping')
            bit_cols = list(col_mapping.columns)
            category_rows = col_mapping.loc[col_mapping.index > 0]
            words = category_rows.to_numpy(dtype=float)
            ordinals = category_rows.index.to_numpy()

            observed = np.full((n_rows, len(bit_cols)), np.nan, dtype=float)
            present_bit_count = sum(bit in X.columns for bit in bit_cols)
            compact_patterns = None
            if present_bit_count == 0 and col in X.columns:
                compact_patterns = self._parse_compact_partial_words(
                    X[col], len(bit_cols), self.base
                )

            for j, bit in enumerate(bit_cols):
                is_invariant = category_rows.shape[0] > 0 and category_rows[bit].nunique(
                    dropna=False
                ) <= 1
                invariant_value = (
                    float(category_rows[bit].iloc[0]) if is_invariant else np.nan
                )

                if compact_patterns is not None:
                    observed[:, j] = compact_patterns[:, j]
                    if is_invariant:
                        observed[np.isnan(observed[:, j]), j] = invariant_value
                    continue

                if bit not in X.columns:
                    # An absent invariant column is a fitted fact.  A absent
                    # varying column is equivalent to erasing every row's bit.
                    if is_invariant:
                        observed[:, j] = invariant_value
                    continue

                raw = X[bit]
                if isinstance(raw, pd.DataFrame):  # defensive: duplicated column name
                    raw = raw.iloc[:, 0]

                raw_values = np.asarray(raw)
                original_na = pd.isna(raw).to_numpy()
                question = np.fromiter(
                    (
                        isinstance(value, str) and value.strip() == '?'
                        for value in raw_values
                    ),
                    dtype=bool,
                    count=len(raw_values),
                )
                numeric = np.array(
                    pd.to_numeric(raw, errors='coerce'), dtype=float, copy=True
                )
                invalid_text = ~original_na & ~question & np.isnan(numeric)
                # -1 is outside every normal Base-N alphabet and therefore makes
                # the row an explicit no_match instead of looking erased.
                numeric[invalid_text] = -1.0
                observed[:, j] = numeric

                if erasure_frame is not None:
                    observed[erasure_frame[bit].to_numpy(dtype=bool), j] = np.nan

                if is_invariant:
                    erased_here = np.isnan(observed[:, j])
                    observed[erased_here, j] = invariant_value

            recovered_ordinals, statuses = self._recover_column(
                observed, words, ordinals, self.base
            )

            ordinal_switch = next(
                item for item in self.ordinal_encoder.mapping if item.get('col') == col
            )
            ordinal_mapping = ordinal_switch.get('mapping')
            inverse = pd.Series(
                data=ordinal_mapping.index, index=ordinal_mapping.array
            )
            decoded = pd.Series(recovered_ordinals, index=X.index).map(inverse)
            try:
                decoded = decoded.astype(ordinal_switch.get('data_type'))
            except (TypeError, ValueError):
                # Unrecovered rows require NaN and may prevent an integer cast.
                pass

            recovered_data[col] = decoded.to_numpy()
            status_data[f'{col}_recovery_status'] = np.asarray(statuses, dtype=object)

        all_bit_cols = [
            bit
            for switch in self.mapping
            for bit in switch.get('mapping').columns
        ]

        # Reconstruct a stable, name-keyed frame.  Assigning NumPy arrays avoids
        # pandas alignment and therefore preserves row order with duplicate index.
        ordered_names: list = []
        for name in getattr(self, 'feature_names_in_', []):
            if name in recovered_data or name in X.columns:
                ordered_names.append(name)
        for name in X.columns:
            if name not in all_bit_cols and name not in ordered_names:
                ordered_names.append(name)
        for name in recovered_data:
            if name not in ordered_names:
                ordered_names.append(name)

        recovered = pd.DataFrame(index=X.index)
        for name in ordered_names:
            if name in recovered_data:
                recovered[name] = recovered_data[name]
            elif name in X.columns and name not in all_bit_cols:
                recovered[name] = X[name].to_numpy()

        status = RecoveryStatusFrame(status_data, index=X.index)

        if boolean_status or status_format in {'bool', 'boolean'}:
            status = status.recovered_flags

        as_df = self.return_df if return_df is None else return_df
        if as_df:
            result = PartialInverseResult(recovered, status)
        else:
            result = PartialInverseResult(recovered.to_numpy(), status.to_numpy())

        if return_status:
            return result
        return result.recovered

    partial_inverse_transform = inverse_transform_partial
    partial_inverse = inverse_transform_partial
    partial_recovery = inverse_transform_partial
    inverse_partial = inverse_transform_partial
    recover = inverse_transform_partial
    recover_partial = inverse_transform_partial
    recover_erased = inverse_transform_partial
    inverse_transform_with_status = inverse_transform_partial

    @staticmethod
    def _parse_compact_partial_words(column, n_bits: int, base: int) -> np.ndarray | None:
        """Parse compact words such as ``"10?"`` when no bit columns are present."""
        if isinstance(column, pd.DataFrame):
            column = column.iloc[:, 0]
        values = np.asarray(column)
        parsed = np.full((len(values), n_bits), np.nan, dtype=float)
        saw_pattern = False

        for i, value in enumerate(values):
            if pd.isna(value):
                continue
            if not isinstance(value, str):
                return None
            word = value.strip()
            if len(word) != n_bits:
                return None
            row = []
            for char in word:
                if char == '?':
                    row.append(np.nan)
                elif char.isdigit() and int(char) < max(base, 2 if base == 1 else base):
                    row.append(float(char))
                    saw_pattern = True
                else:
                    return None
            parsed[i, :] = row
            saw_pattern = True

        return parsed if saw_pattern else None

    def _coerce_partial_erasure(self, erasure, X: pd.DataFrame) -> pd.DataFrame | None:
        """Normalize a user-supplied erasure mask over all fitted bit columns."""
        if erasure is None:
            return None

        all_bits = [
            bit
            for switch in self.mapping
            for bit in switch.get('mapping').columns
        ]
        n_rows = X.shape[0]
        frame = pd.DataFrame(False, index=X.index, columns=all_bits)

        if isinstance(erasure, pd.DataFrame):
            if erasure.shape[0] != n_rows:
                raise ValueError(
                    f'erasure mask has {erasure.shape[0]} rows but X_in has {n_rows}'
                )
            for column in erasure.columns:
                if column in all_bits:
                    frame[column] = erasure[column].to_numpy(dtype=bool)
                elif column not in X.columns:
                    raise ValueError(f'erasure mask references unknown column {column!r}')
            return frame

        array = np.asarray(erasure, dtype=bool)
        present_bits = [column for column in X.columns if column in all_bits]

        if array.ndim == 0:
            frame[all_bits] = bool(array)
            return frame

        if array.ndim == 1:
            if array.shape[0] == len(present_bits):
                targets = present_bits
            elif array.shape[0] == len(all_bits):
                targets = all_bits
            elif array.shape[0] == n_rows:
                frame.loc[:, :] = array[:, None]
                return frame
            else:
                raise ValueError(
                    '1-D erasure mask must contain one value per present bit, '
                    'fitted bit, or input row'
                )
            for bit, flag in zip(targets, array, strict=False):
                frame[bit] = bool(flag)
            return frame

        if array.ndim != 2:
            raise ValueError('erasure must be a boolean array-like or DataFrame')
        if array.shape[0] != n_rows:
            raise ValueError(
                f'erasure mask has {array.shape[0]} rows but X_in has {n_rows}'
            )

        if array.shape[1] == len(present_bits):
            targets = present_bits
        elif array.shape[1] == len(all_bits):
            targets = all_bits
        elif array.shape[1] == X.shape[1]:
            for j, column in enumerate(X.columns):
                if column in all_bits:
                    frame[column] = array[:, j]
            return frame
        else:
            raise ValueError(
                '2-D erasure mask must align with the present bit columns, all '
                'fitted bit columns, or all input columns'
            )

        for j, bit in enumerate(targets):
            frame[bit] = array[:, j]
        return frame

    @staticmethod
    def _build_partial_lookup(mask, words: np.ndarray, labels: np.ndarray) -> dict:
        """Prepare one surviving-bit lookup for a fixed erasure mask."""
        known_positions = [j for j, erased in enumerate(mask) if not erased]
        lookup: dict[tuple[int, ...], object] = {}
        for word, label in zip(words, labels, strict=False):
            key = tuple(int(word[j]) for j in known_positions)
            if key in lookup:
                lookup[key] = _AMBIGUOUS_MATCH
            else:
                lookup[key] = label
        return lookup

    @classmethod
    def _recover_column(
        cls,
        observed: np.ndarray,
        words: np.ndarray,
        labels: np.ndarray,
        base: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover one field, preparing each distinct mask lookup only once."""
        n_rows = observed.shape[0]
        erased = np.isnan(observed)

        # Base-1 output is the library's one-hot-like 0/1 representation.
        valid_digits = np.array([0, 1]) if base == 1 else np.arange(base)
        invalid_row = (~erased & ~np.isin(observed, valid_digits)).any(axis=1)

        recovered = np.full(n_rows, np.nan, dtype=float)
        status = np.full(n_rows, NO_MATCH, dtype=object)

        if n_rows == 0:
            return recovered, status

        for mask_array in np.unique(erased, axis=0):
            mask = tuple(bool(value) for value in mask_array)
            row_indices = np.flatnonzero((erased == mask_array).all(axis=1))
            lookup = cls._build_partial_lookup(mask, words, labels)
            known_positions = [j for j, is_erased in enumerate(mask) if not is_erased]

            for row_index in row_indices:
                if invalid_row[row_index]:
                    continue
                key = tuple(int(observed[row_index, j]) for j in known_positions)
                match = lookup.get(key)
                if match is None:
                    continue
                if match is _AMBIGUOUS_MATCH:
                    status[row_index] = AMBIGUOUS
                else:
                    recovered[row_index] = match
                    status[row_index] = RECOVERED

        return recovered, status

    def calc_required_digits(self, values: list) -> int:
        """Figure out how many digits we need to represent the classes present.

        Parameters
        ----------
        values: list
            list of values.

        Returns
        -------
        int
            number of digits necessary for encoding.
        """
        if self.base == 1:
            digits = len(values) + 1
        else:
            digits = _ceillogint(len(values) + 1, self.base)

        return digits

    def basen_encode(self, X_in: pd.DataFrame, cols=None):
        """
        Basen encoding encodes the integers as basen code with one column per digit.

        Parameters
        ----------
        X_in: DataFrame
        cols: list-like, default None
            Column names in the DataFrame to be encoded

        Returns
        -------
        dummies : DataFrame

        """
        X = X_in.copy(deep=True)

        cols = X.columns.tolist()

        for switch in self.mapping:
            col = switch.get('col')
            mod = switch.get('mapping')

            base_df = mod.reindex(X[col])
            base_df = base_df.set_index(X.index)
            X = pd.concat([base_df, X], axis=1)

            old_column_index = cols.index(col)
            cols[old_column_index : old_column_index + 1] = mod.columns

        return X.reindex(columns=cols)

    def basen_to_integer(self, X: pd.DataFrame, cols, base):
        """
        Convert basen code as integers.

        Parameters
        ----------
        X : DataFrame
            encoded data
        cols : list-like
            Column names in the DataFrame that be encoded
        base : int
            The base of transform

        Returns
        -------
        numerical: DataFrame

        """
        out_cols = X.columns.tolist()

        for col in cols:
            col_list = [
                col0 for col0 in out_cols if re.match(re.escape(str(col)) + '_\\d+', str(col0))
            ]
            insert_at = out_cols.index(col_list[0])

            if base == 1:
                value_array = np.array([int(col0.split('_')[-1]) for col0 in col_list])
            else:
                len0 = len(col_list)
                value_array = np.array([base ** (len0 - 1 - i) for i in range(len0)])
            X.insert(insert_at, col, np.dot(X[col_list].values, value_array.T))
            X = X.drop(col_list, axis=1)
            out_cols = X.columns.tolist()

        return X

    def col_transform(self, col, digits):
        """The lambda body to transform the column values."""
        if col is None or float(col) < 0.0:
            return None
        else:
            col = self.number_to_base(int(col), self.base, digits)
            if len(col) == digits:
                return col
            else:
                return [0 for _ in range(digits - len(col))] + col

    @staticmethod
    def number_to_base(n: int, b: int, limit: int) -> list[int]:
        """Convert number to base n representation (as list of digits).

        The list will be of length `limit`.

        Parameters
        ----------
        n: int
            number to convert
        b: int
            base
        limit: int
            length of representation.

        Returns
        -------
        list[int]
            base n representation as list of length limit containing the digits.
        """
        if b == 1:
            return [0 if n != _ else 1 for _ in range(limit)]

        if n == 0:
            return [0 for _ in range(limit)]

        digits = []
        for _ in range(limit):
            digits.append(int(n % b))
            n, _ = divmod(n, b)

        return digits[::-1]
