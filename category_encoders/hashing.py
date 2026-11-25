"""The hashing module contains all methods and classes related to the hashing trick."""

import hashlib
import math
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import sparse

import category_encoders.utils as util

__author__ = 'willmcginnis', 'LiuShulun'

# Bootstrap seed of the signed sparse mode. Using a fixed constant (instead of
# drawing entropy at construction time) makes the bucket/sign mapping reproducible
# across calls, processes and machines without the caller pinning anything.
DEFAULT_SIGNED_HASH_SEED = 23

# Separator placed between the decimal seed and the value before hashing. A NUL
# byte cannot appear in the decimal rendering of the seed, so the seed and value
# parts can never be confused even when the value itself contains digits.
_SIGNED_HASH_SEED_SEPARATOR = b'\x00'


class HashingEncoder( util.UnsupervisedTransformerMixin,util.BaseEncoder):
    """A multivariate hashing implementation with configurable dimensionality/precision.

    The advantage of this encoder is that it does not maintain a dictionary of observed categories.
    Consequently, the encoder does not grow in size and accepts new values during data scoring
    by design.

    It's important to read about how max_process & max_sample work
    before setting them manually, inappropriate setting slows down encoding.

    Default value of 'max_process' is 1 on Windows because multiprocessing might cause issues,
    see in :
    https://github.com/scikit-learn-contrib/categorical-encoding/issues/215
    https://docs.python.org/2/library/multiprocessing.html?highlight=process#windows

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
    hash_method: str
        which hashing method to use. Any method from hashlib works.
    max_process: int
        how many processes to use in transform(). Limited in range(1, 64).
        By default, it uses half of the logical CPUs.
        For example, 4C4T makes max_process=2, 4C8T makes max_process=4.
        Set it larger if you have a strong CPU.
        It is not recommended to set it larger than is the count of the
        logical CPUs as it will actually slow down the encoding.
    max_sample: int
        how many samples to encode by each process at a time.
        This setting is useful on low memory machines.
        By default, max_sample=(all samples num)/(max_process).
        For example, 4C8T CPU with 100,000 samples makes max_sample=25,000,
        6C12T CPU with 100,000 samples makes max_sample=16,666.
        It is not recommended to set it larger than the default value.
    n_components: int
        the number of output features (hash buckets) per encoded column.
        Defaults to 8. This is the *dimension*, not a bit count —
        ``n_components=32`` produces 32 buckets, not 2**32. The collision
        rate is roughly ``1 / n_components``; prefer larger values for
        high-cardinality features. See `Feature hashing on Wikipedia
        <https://en.wikipedia.org/wiki/Feature_hashing>`_.
    process_creation_method: string
        either "fork", "spawn" or "forkserver" (availability depends on your
        platform). See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        for more details and tradeoffs. Defaults to "fork" on linux/macos as it
        is the fastest option and to "spawn" on windows as it is the only one
        available
    sparse_output: bool
        when False (the default) the encoder keeps its historical behaviour and returns
        a dense, unsigned count matrix (a DataFrame, or a numpy array when
        ``return_df=False``). When True, ``transform`` returns a *signed*
        :class:`scipy.sparse.csr_matrix` built directly from per-hit coordinates:
        every value is mapped to exactly one bucket and contributes ``+1`` or ``-1``
        to it, so colliding values with opposite signs cancel instead of piling up.
        Buckets whose contributions cancel to zero are removed from the sparse
        storage, and no ``n_samples * n_components`` dense array is ever allocated.
        Non-encoded numeric columns are appended on the right, matching the dense
        column order. The dense unsigned path is untouched by this option.
    random_state: int, numpy RandomState or None
        pins the seed of the signed mapping used when ``sparse_output=True``.
        ``None`` (the default) resolves to a fixed seed, so output is reproducible
        across calls, processes and machines without having to pass anything. The
        bucket and sign of a value depend only on the value and this seed, never on
        its row position or on how rows are split across worker processes. Ignored
        by the dense unsigned path.

    Notes
    -----
    The signed sparse mode follows the classic "hashing trick with sign": for a
    value ``v`` the digest is ``h = int(hash("<seed>\\x00<v>"))`` (little-endian),
    the bucket is ``h % N`` and the sign is ``+1`` when ``(h // N) % 2 == 0`` and
    ``-1`` otherwise. The same computation is exposed for a single chunk through
    :meth:`HashingEncoder.hash_chunk` (pass ``signed=True``) and
    :meth:`HashingEncoder.hash_chunk_signed`. The seed defaults to
    :data:`DEFAULT_SIGNED_HASH_SEED`, so the mapping is reproducible without
    passing ``random_state``.

    Example
    -------
    >>> from category_encoders.hashing import HashingEncoder
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
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> y = bunch.target
    >>> he = HashingEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
    >>> numeric_dataset = he.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 13 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   col_0        1460 non-null   int64
     1   col_1        1460 non-null   int64
     2   col_2        1460 non-null   int64
     3   col_3        1460 non-null   int64
     4   col_4        1460 non-null   int64
     5   col_5        1460 non-null   int64
     6   col_6        1460 non-null   int64
     7   col_7        1460 non-null   int64
     8   Id           1460 non-null   float64
     9   MSSubClass   1460 non-null   float64
     10  MSZoning     1460 non-null   object
     11  LotFrontage  1201 non-null   float64
     12  YearBuilt    1460 non-null   float64
    dtypes: float64(4), int64(8), object(1)
    memory usage: 148.4+ KB
    None

    For a very wide hashing space where only a few buckets fire per row, request the
    optional signed sparse output to skip the full dense matrix altogether:

    >>> wide = HashingEncoder(n_components=100000, sparse_output=True, random_state=0)
    >>> matrix = wide.fit_transform(X[['CentralAir', 'Heating']])
    >>> matrix.shape
    (1460, 100000)
    >>> matrix.nnz <= 2 * X.shape[0]
    True

    References
    ----------
    .. [1] Feature Hashing for Large Scale Multitask Learning, from
    https://alex.smola.org/papers/2009/Weinbergeretal09.pdf
    .. [2] Don't be tricked by the Hashing Trick, from
    https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087

    """

    prefit_ordinal = False
    encoding_relation = util.EncodingRelation.ONE_TO_M
    _VALID_HANDLE_MISSING = None
    _VALID_HANDLE_UNKNOWN = None

    def __init__(
        self,
        max_process=0,
        max_sample=0,
        verbose=0,
        n_components=8,
        cols=None,
        drop_invariant=False,
        return_df=True,
        hash_method='md5',
        process_creation_method='fork',
        sparse_output=False,
        random_state=None,
        min_group_size: int | float | None = None,
        min_group_name: str | None = None,
        combine_min_nan_groups: bool | str | None = None,
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown='does not apply',
            handle_missing='does not apply',
            min_group_size=min_group_size,
            min_group_name=min_group_name,
            combine_min_nan_groups=combine_min_nan_groups,
        )

        if max_process not in range(1, 128):
            if platform.system() == 'Windows':
                self.max_process = 1
            else:
                self.max_process = int(math.ceil(multiprocessing.cpu_count() / 2))
                if self.max_process < 1:
                    self.max_process = 1
                elif self.max_process > 128:
                    self.max_process = 128
        else:
            self.max_process = max_process
        self.max_sample = int(max_sample)
        if platform.system() == 'Windows':
            self.process_creation_method = 'spawn'
        else:
            self.process_creation_method = process_creation_method
        self.data_lines = 0
        self.X = None

        self.n_components = n_components
        self.hash_method = hash_method
        self.sparse_output = sparse_output
        self.random_state = random_state
        if self.sparse_output:
            # A CSR matrix cannot be wrapped into a pandas DataFrame, so disable the
            # sklearn output wrapping for this estimator. Without this a global
            # sklearn config ``transform_output='pandas'`` would raise when a sparse
            # encoder is used inside a pipeline.
            self._sklearn_output_config = {'transform': 'default'}
        # Resolve the seed once to a plain integer. Worker processes receive this
        # integer and repeated transforms reuse it, so the signed bucket/sign
        # mapping is identical regardless of process or chunk boundaries.
        self.signed_hash_seed_ = self._resolve_seed(random_state)

    @staticmethod
    def _resolve_seed(random_state) -> int:
        """Turn ``random_state`` into the fixed integer seed of the signed path.

        ``None`` maps to :data:`DEFAULT_SIGNED_HASH_SEED` so the default signed
        output is reproducible without pinning anything. Integers pass through
        unchanged (so ``random_state=7`` and ``seed=7`` agree), while a
        ``RandomState``/``Generator`` is drawn from once, deterministically given
        its state.
        """
        if random_state is None:
            return DEFAULT_SIGNED_HASH_SEED
        if isinstance(random_state, (int, np.integer)) and not isinstance(random_state, bool):
            return int(random_state)
        if isinstance(random_state, np.random.Generator):
            return int(random_state.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        # numpy RandomState (and anything else exposing randint)
        return int(random_state.randint(0, np.iinfo(np.int64).max, dtype=np.int64))

    def _fit(self, X, y=None, **kwargs):
        pass

    def fit(self, X, y=None, **kwargs):
        """Fit the encoder.

        The default (dense) path is unchanged. With ``sparse_output=True`` the
        fitted attributes (``feature_names_out_``, ``invariant_cols``) are learned
        from the signed CSR representation directly, so a very wide ``n_components``
        never triggers a ``n_samples * n_components`` dense allocation, not even
        during fit.
        """
        if not self.sparse_output:
            return super().fit(X, y, **kwargs)
        return self._fit_sparse(X, y, **kwargs)

    def _fit_sparse(self, X, y=None, **kwargs):
        """Fit bookkeeping for the signed sparse path without densifying."""
        X = util.convert_input(X)
        self._check_fit_inputs(X, y)
        self._validate_handle_strategies()
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.feature_dtypes_in_ = list(X.dtypes)
        self._dim = X.shape[1]
        self._determine_fit_columns(X)

        if not set(self.cols).issubset(X.columns):
            missing = [col for col in self.cols if col not in X.columns]
            raise ValueError(f'X does not contain the columns listed in cols: {missing}')

        if self.handle_missing == 'error' and X[self.cols].isna().any().any():
            raise ValueError('Columns to be encoded cannot contain null')

        self.min_group_lumping_ = self._fit_min_group_lumping(X)
        if self.min_group_lumping_:
            X = X.copy(deep=True)
            self._apply_min_group_lumping(X)

        self._fit(X, y, **kwargs)

        if not list(self.cols):
            # Nothing to hash: the dense path returns the input untouched.
            self.invariant_cols = []
            self._output_names_full_ = list(X.columns)
            self.feature_names_out_ = np.asarray(list(X.columns), dtype=object)
            return self

        hashed = self.hashing_trick(
            X, hashing_method=self.hash_method, N=self.n_components, cols=self.cols
        )
        # Output column order matches the dense path: hash buckets then pass-through.
        names = [f'col_{d}' for d in range(self.n_components)] + [
            col for col in X.columns if col not in self.cols
        ]
        self._output_names_full_ = names

        self.invariant_cols = []
        if self.drop_invariant:
            invariant_positions = self._invariant_sparse_positions(hashed, len(names))
            self.invariant_cols = [names[i] for i in invariant_positions]
        self.feature_names_out_ = np.asarray(
            [name for name in names if name not in self.invariant_cols], dtype=object
        )
        return self

    @staticmethod
    def _invariant_sparse_positions(matrix: sparse.csr_matrix, n_columns: int) -> list[int]:
        """Positions of columns that hold one value across all rows, without densifying.

        A column is invariant when every row has the same value. For a sparse matrix
        that means either the column has no stored entry at all (all rows are zero),
        or every row stores the very same single non-zero value. The check reads the
        CSR structure directly, so it stays linear in the number of stored entries.
        """
        coo = matrix.tocoo()
        n_rows = matrix.shape[0]
        if n_rows == 0:
            return list(range(n_columns))

        counts = np.bincount(coo.col, minlength=n_columns)
        # np.minimum.at / maximum.at collapse each column to its extreme value; an
        # invariant column has min == max and is stored in every row.
        minima = np.full(n_columns, np.inf)
        maxima = np.full(n_columns, -np.inf)
        np.minimum.at(minima, coo.col, coo.data)
        np.maximum.at(maxima, coo.col, coo.data)

        invariants = []
        for position in range(n_columns):
            if counts[position] == 0:
                invariants.append(position)  # column is all zeros
            elif counts[position] == n_rows and minima[position] == maxima[position]:
                invariants.append(position)  # every row holds the same value
        return invariants

    def _drop_invariants(self, df, override_return_df: bool):
        """Drop zero-variance columns from the sparse output and hand it back."""
        if not sparse.issparse(df):
            return super()._drop_invariants(df, override_return_df)
        if self.drop_invariant and self.invariant_cols:
            full_names = getattr(self, '_output_names_full_', [])
            keep = [i for i, name in enumerate(full_names) if name not in self.invariant_cols]
            df = df[:, keep]
        return df

    def _transform(self, X, override_return_df=False):
        """Perform the transformation to new categorical data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)

        # the shared validation in _check_transform_inputs (including the DataFrame
        # superset rule, GH #367) has already run in transform()

        if not list(self.cols):
            return X

        X = self.hashing_trick(
            X,
            hashing_method=self.hash_method,
            N=self.n_components,
            cols=self.cols,
        )

        return X

    @staticmethod
    def hash_chunk(
        hash_method: str,
        np_df: np.ndarray,
        N: int,
        *,
        signed: bool = False,
        seed: int = DEFAULT_SIGNED_HASH_SEED,
    ) -> np.ndarray | sparse.csr_matrix:
        """Perform hashing on the given numpy array.

        Parameters
        ----------
        hash_method: str
            Hashlib method to use.
        np_df: np.ndarray
            Data to hash.
        N: int
            Number of bits to encode the data.
        signed: bool
            when False (the default) the historical dense, unsigned count matrix is
            returned, exactly as before. When True this is a shortcut for
            :meth:`hash_chunk_signed`: a signed CSR matrix is returned and no
            ``n_rows * N`` dense array is allocated.
        seed: int
            seed of the signed mapping. Only used when ``signed=True``.

        Returns
        -------
        np.ndarray or scipy.sparse.csr_matrix
            Hashed data. Dense ``int`` ndarray by default, signed ``csr_matrix``
            when ``signed=True``.
        """
        if signed:
            return HashingEncoder.hash_chunk_signed(hash_method, np_df, N, seed)

        # Calling getattr outside the loop saves some time in the loop
        hasher_constructor = getattr(hashlib, hash_method)
        # Same when the call to getattr is implicit
        int_from_bytes = int.from_bytes
        result = np.zeros((np_df.shape[0], N), dtype='int')
        for i, row in enumerate(np_df):
            for val in row:
                if val is not None:
                    hasher = hasher_constructor()
                    # Computes an integer index from the hasher digest. The endianness is
                    # "big" as the code used to read:
                    # column_index = int(hasher.hexdigest(), 16) % N
                    # which is implicitly considering the hexdigest to be big-endian,
                    # even if the system is little-endian.
                    # Building the index that way is about 30% faster than using the
                    # hexdigest.
                    hasher.update(bytes(str(val), 'utf-8'))
                    column_index = int_from_bytes(hasher.digest(), byteorder='big') % N
                    result[i, column_index] += 1
        return result

    @staticmethod
    def hash_chunk_signed(
        hash_method: str, np_df: np.ndarray, N: int, seed: int = DEFAULT_SIGNED_HASH_SEED
    ) -> sparse.csr_matrix:
        """Signed feature hashing of one chunk, returned as a CSR matrix.

        Every non-null value maps to exactly one bucket and contributes ``+1`` or
        ``-1``. The bucket is the digest modulo ``N`` and the sign is taken from the
        next digest "digit" (``(digest // N) % 2``); this is the classical signed
        hashing trick, which makes colliding features cancel rather than pile up.

        Hits are accumulated as sparse ``(row, column)`` coordinates, so a
        ``n_rows * N`` dense array is never allocated. Collisions are summed by
        :meth:`scipy.sparse.coo_matrix.sum_duplicates` and buckets whose positive and
        negative contributions cancel to exactly zero are removed by
        ``eliminate_zeros``.

        The mapping depends only on ``(seed, value)``: not on the row position, on
        how the input is split into chunks, nor on the process executing the chunk.

        Parameters
        ----------
        hash_method: str
            Hashlib method to use.
        np_df: np.ndarray
            Data to hash.
        N: int
            Number of hash buckets.
        seed: int
            Seed pinning the bucket/sign mapping. Defaults to
            :data:`DEFAULT_SIGNED_HASH_SEED`.

        Returns
        -------
        scipy.sparse.csr_matrix
            Signed hashed data with shape ``(np_df.shape[0], N)`` and integer values.
        """
        if N < 1:
            raise ValueError(f'N (the number of hash buckets) must be at least 1, got {N}.')

        hasher_constructor = getattr(hashlib, hash_method)
        int_from_bytes = int.from_bytes
        # The seed is folded into every value as "<seed>\x00<value>" before hashing.
        # A NUL byte cannot occur in the decimal seed, so the two parts never merge,
        # while the combined key stays a pure function of (seed, value): independent
        # of row position, chunk boundaries and the executing process.
        seed_prefix = str(int(seed)).encode('utf-8') + _SIGNED_HASH_SEED_SEPARATOR

        row_indices = []
        column_indices = []
        values = []
        for i, row in enumerate(np_df):
            for val in row:
                if val is not None:
                    hasher = hasher_constructor()
                    hasher.update(seed_prefix)
                    hasher.update(bytes(str(val), 'utf-8'))
                    # Little-endian byte order matches the historical reference of the
                    # signed mapping; the bucket is the digest modulo N and the sign is
                    # read from the next base-N "digit". Together they are the classical
                    # signed hashing trick: collisions cancel instead of piling up.
                    digest_int = int_from_bytes(hasher.digest(), byteorder='little')
                    column_index = digest_int % N
                    sign = 1 if (digest_int // N) % 2 == 0 else -1
                    row_indices.append(i)
                    column_indices.append(column_index)
                    values.append(sign)

        result = sparse.coo_matrix(
            (
                np.asarray(values, dtype=np.int64),
                (
                    np.asarray(row_indices, dtype=np.int64),
                    np.asarray(column_indices, dtype=np.int64),
                ),
            ),
            shape=(np_df.shape[0], N),
        ).tocsr()
        # Sum repeated (row, bucket) hits and drop entries whose opposite-sign
        # contributions cancelled to zero. Neither step materialises a dense block.
        result.sum_duplicates()
        result.eliminate_zeros()
        return result

    def _hash_parallel(self, df: pd.DataFrame, N: int) -> sparse.csr_matrix:
        """Signed hashing trick in parallel, preserving the original row order."""
        np_df = df.to_numpy()
        if np_df.shape[0] == 0:
            return sparse.csr_matrix((0, N), dtype=np.int64)
        ctx = multiprocessing.get_context(self.process_creation_method)

        with ProcessPoolExecutor(max_workers=self.max_process, mp_context=ctx) as executor:
            chunks = list(
                executor.map(
                    self.hash_chunk_signed,
                    [self.hash_method] * self.max_process,
                    np.array_split(np_df, self.max_process),
                    [N] * self.max_process,
                    [self.signed_hash_seed_] * self.max_process,
                )
            )

        # Each chunk is (rows_in_chunk, N); vstacking in split order restores the
        # global row positions exactly as in the input.
        result = sparse.vstack(chunks, format='csr')
        result.sum_duplicates()
        result.eliminate_zeros()
        return result

    def _hash_no_parallel(self, df: pd.DataFrame, N: int) -> sparse.csr_matrix:
        """Signed hashing trick in a single process."""
        np_df = df.to_numpy()
        return self.hash_chunk_signed(self.hash_method, np_df, N, self.signed_hash_seed_)


    def hashing_trick_with_np_parallel(self, df: pd.DataFrame, N: int) -> pd.DataFrame:
        """Perform the hashing trick in parallel.

        Parameters
        ----------
        df: pd.DataFrame
           data to hash.
        N: int
           how many bits to use to represent the feature.

        Returns
        -------
        pd.DataFrame
           hashed data.
        """
        np_df = df.to_numpy()
        ctx = multiprocessing.get_context(self.process_creation_method)

        with ProcessPoolExecutor(max_workers=self.max_process, mp_context=ctx) as executor:
            result = np.concatenate(
                list(
                    executor.map(
                        self.hash_chunk,
                        [self.hash_method] * self.max_process,
                        np.array_split(np_df, self.max_process),
                        [N] * self.max_process,
                    )
                )
            )

        return pd.DataFrame(result, index=df.index)

    def hashing_trick_with_np_no_parallel(self, df: pd.DataFrame, N: int) -> pd.DataFrame:
        """Perform the hashing trick in a single thread (non-parallel).

        Parameters
        ----------
        df: pd.DataFrame
           data to hash.
        N: int
           how many bits to use to represent the feature.

        Returns
        -------
        pd.DataFrame
           hashed data.
        """
        np_df = df.to_numpy()

        result = HashingEncoder.hash_chunk(self.hash_method, np_df, N)

        return pd.DataFrame(result, index=df.index)

    def hashing_trick(self, X_in, hashing_method='md5', N=2, cols=None, make_copy=False):
        """A basic hashing implementation with configurable dimensionality/precision.

        Performs the hashing trick on a pandas dataframe, `X`, using the hashing method from
        hashlib identified by `hashing_method`.
        The number of output dimensions (`N`), and columns to hash (`cols`) are also configurable.

        Parameters
        ----------
        X_in: pandas dataframe
            description text
        hashing_method: string, optional
            description text
        N: int, optional
            description text
        cols: list, optional
            description text
        make_copy: bool, optional
            description text

        Returns
        -------
        out : pandas.DataFrame or scipy.sparse.csr_matrix
            A hashing encoded dataframe, or a signed CSR matrix when
            ``sparse_output=True``.

        References
        ----------
        Cite the relevant literature, e.g. [1]_.  You may also cite these
        references in the notes section above.
        .. [1] Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola;
        Josh Attenberg (2009). Feature Hashing for Large Scale Multitask Learning. Proc. ICML.

        """
        if hashing_method not in hashlib.algorithms_available:
            raise ValueError(
                f"Hashing Method: {hashing_method} not available. "
                f"Please use one of: {', '.join([str(x) for x in hashlib.algorithms_available])}"
            )

        if make_copy:
            X = X_in.copy(deep=True)
        else:
            X = X_in

        if cols is None:
            cols = X.columns

        new_cols = [f'col_{d}' for d in range(N)]

        X_cat = X.loc[:, cols]
        X_num = X.loc[:, [x for x in X.columns if x not in cols]]

        if self.sparse_output:
            if self.max_process == 1:
                X_cat = self._hash_no_parallel(X_cat, N)
            else:
                X_cat = self._hash_parallel(X_cat, N)

            if X_num.shape[1]:
                # Append untouched pass-through columns on the right, matching the
                # dense path's [hashed columns, remaining columns] order. A sparse
                # matrix can only store numeric values, so non-numeric pass-through
                # columns are rejected with a clear message instead of silently
                # corrupting the output.
                non_numeric = [
                    col for col in X_num.columns if not is_numeric_dtype(X_num[col])
                ]
                if non_numeric:
                    raise ValueError(
                        'sparse_output=True can only pass through numeric columns; '
                        f'the following columns to keep were not hashed and are non-numeric: '
                        f'{non_numeric}. Either add them to `cols`, set cols=None, or set '
                        'sparse_output=False.'
                    )
                X_num_sparse = sparse.csr_matrix(X_num.to_numpy(dtype=np.float64))
                return sparse.hstack([X_cat, X_num_sparse], format='csr')
            return X_cat

        if self.max_process == 1:
            X_cat = self.hashing_trick_with_np_no_parallel(X_cat, N)
        else:
            X_cat = self.hashing_trick_with_np_parallel(X_cat, N)

        X_cat.columns = new_cols

        X = pd.concat([X_cat, X_num], axis=1)

        return X
