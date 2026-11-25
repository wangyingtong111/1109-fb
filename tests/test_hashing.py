"""Tests for the HashingEncoder."""
import hashlib
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal


class TestHashingEncoder(TestCase):
    """Tests for the HashingEncoder."""

    def test_must_not_reset_index(self):
        """Test that the HashingEncoder does not reset the index."""
        columns = ['column1', 'column2', 'column3', 'column4']
        df = pd.DataFrame([[i, i, i, i] for i in range(10)], columns=columns)
        df = df.iloc[2:8, :]
        target_columns = ['column1', 'column2', 'column3']

        single_process_encoder = encoders.HashingEncoder(max_process=1, cols=target_columns)
        single_process_encoder.fit(df, None)
        df_encoded_single_process = single_process_encoder.transform(df)
        assert_index_equal(df.index, df_encoded_single_process.index)
        self.assertEqual(df.shape[0],
                         pd.concat([df, df_encoded_single_process], axis=1).shape[0])

        multi_process_encoder = encoders.HashingEncoder(cols=target_columns)
        multi_process_encoder.fit(df, None)
        df_encoded_multi_process = multi_process_encoder.transform(df)
        assert_index_equal(df.index, df_encoded_multi_process.index)
        self.assertEqual(df.shape[0] , pd.concat([df, df_encoded_multi_process], axis=1).shape[0])

        assert_frame_equal(df_encoded_single_process, df_encoded_multi_process)

    def test_transform_works_with_single_row_df(self):
        """Test that the HashingEncoder works with a single row DataFrame."""
        columns = ['column1', 'column2', 'column3', 'column4']
        df = pd.DataFrame([[i, i, i, i] for i in range(10)], columns=columns)
        df = df.iloc[2:8, :]
        target_columns = ['column1', 'column2', 'column3']

        multi_process_encoder = encoders.HashingEncoder(cols=target_columns)
        multi_process_encoder.fit(df, None)
        df_encoded_multi_process = multi_process_encoder.transform(df.sample(1))

        self.assertEqual(
            multi_process_encoder.n_components + len(list(set(columns) - set(target_columns))),
            df_encoded_multi_process.shape[1]
        )

    def test_simple_example(self):
        """Test the HashingEncoder with a simple example."""
        df = pd.DataFrame(
            {
                'strings': ['aaaa', 'bbbb', 'cccc'],
                'more_strings': ['aaaa', 'dddd', 'eeee'],
            }
        )
        encoder = encoders.HashingEncoder(n_components=4, max_process=2)
        encoder.fit(df)
        expected_df = pd.DataFrame(
            {'col_0': [0, 1, 1], 'col_1': [2, 0, 1], 'col_2': [0, 1, 0], 'col_3': [0, 0, 0]}
        )
        pd.testing.assert_frame_equal(encoder.transform(df), expected_df)


class TestHashingHardening(TestCase):
    """Hardening tests for HashingEncoder mutation survivors."""

    @staticmethod
    def _bucket(value, n_components=8, method='md5'):
        """Independent hashlib reference for the bucket a value lands in."""
        digest = hashlib.new(method, str(value).encode('utf-8'))
        return int(digest.hexdigest(), 16) % n_components

    def test_duplicate_values_within_a_row_accumulate(self):
        """Two identical values in one row increment the same bucket twice.

        Mutant category_encoders/hashing.py::hash_chunk mutmut_34
        (counts += 1 -> = 1) collapses duplicates to a single hit.
        """
        df = pd.DataFrame({'a': ['x', 'x'], 'b': ['x', None]})
        encoder = encoders.HashingEncoder(n_components=8)
        encoder.fit(df)
        out = encoder.transform(df)
        bucket = self._bucket('x')
        self.assertEqual(out.iloc[0, bucket], 2)
        self.assertEqual(out.iloc[0].sum(), 2)

    def test_none_values_are_skipped(self):
        """None entries are not hashed; only real values hit buckets.

        Mutant category_encoders/hashing.py::hash_chunk mutmut_16
        (if val is not None -> if val is None) hashes None instead of the
        present values, leaving the value's bucket cold.
        """
        df = pd.DataFrame({'a': ['x', 'x'], 'b': [None, None]})
        encoder = encoders.HashingEncoder(n_components=8)
        encoder.fit(df)
        out = encoder.transform(df)
        bucket = self._bucket('x')
        self.assertEqual(out.iloc[0, bucket], 1)
        self.assertEqual(out.iloc[1, bucket], 1)
        self.assertEqual(out.iloc[0].sum(), 1)

    def test_hash_method_is_respected(self):
        """The configured hash_method must drive the hashing.

        Mutant category_encoders/hashing.py::_transform mutmut_17 drops the
        hashing_method keyword, silently falling back to the md5 default.
        """
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        md5_out = (
            encoders.HashingEncoder(n_components=8, hash_method='md5')
            .fit(df)
            .transform(df)
        )
        sha_out = (
            encoders.HashingEncoder(n_components=8, hash_method='sha256')
            .fit(df)
            .transform(df)
        )
        self.assertFalse(np.array_equal(md5_out.to_numpy(), sha_out.to_numpy()))

    def test_default_construction_attributes(self):
        """Documented constructor defaults survive mutation.

        Mutants dropping super().__init__ keywords (handle_unknown /
        handle_missing) or flipping defaults (n_components, hash_method,
        verbose) are visible on these attributes.
        """
        encoder = encoders.HashingEncoder()
        self.assertEqual(encoder.n_components, 8)
        self.assertEqual(encoder.hash_method, 'md5')
        self.assertEqual(encoder.verbose, 0)
        self.assertEqual(encoder.handle_unknown, 'does not apply')
        self.assertEqual(encoder.handle_missing, 'does not apply')

    def test_hashing_trick_hashes_all_columns_by_default(self):
        """hashing_trick with cols=None hashes every column.

        Mutant category_encoders/hashing.py::hashing_trick mutmut_15
        (cols = X.columns -> cols = None) selects no columns.
        """
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['p', 'q']})
        trick = encoders.HashingEncoder().hashing_trick(df, N=3)
        self.assertEqual(trick.shape, (2, 3))
        self.assertEqual(trick.iloc[0].sum(), 2)


def _signed_reference(value, n_components, seed=23, method='md5'):
    """Independent hashlib reference for the signed bucket and sign of a value."""
    number = int.from_bytes(
        hashlib.new(method, (str(seed) + '\0' + str(value)).encode('utf-8')).digest(), 'little'
    )
    bucket = number % n_components
    sign = 1 if (number // n_components) % 2 == 0 else -1
    return bucket, sign


def _find_signed_collision_pair(n_components=16):
    """Return ``(old, other, separate)`` tokens for the cancellation scenario.

    ``old`` and ``other`` share a bucket but carry opposite signs; ``separate`` maps to
    a different bucket.
    """
    tokens = [f'name{i}' for i in range(2000)]
    located = {token: _signed_reference(token, n_components) for token in tokens}
    old = next(
        a
        for a in tokens
        if any(
            located[b][0] == located[a][0] and located[b][1] != located[a][1] for b in tokens
        )
    )
    other = next(
        b
        for b in tokens
        if located[b][0] == located[old][0] and located[b][1] != located[old][1]
    )
    separate = next(t for t in tokens if located[t][0] != located[old][0])
    return old, other, separate


class TestHashingSignedSparse(TestCase):
    """Signed CSR output: signs, collision cancellation, seeds, and no densifying."""

    def test_single_input_maps_to_one_signed_bucket(self):
        """A single non-null value produces exactly one stored entry with its sign."""
        arr = encoders.HashingEncoder.hash_chunk(
            'md5', np.array([['solo']], dtype=object), 128, signed=True
        )
        self.assertTrue(hasattr(arr, 'nnz'))
        self.assertEqual(arr.nnz, 1)
        bucket, sign = _signed_reference('solo', 128)
        self.assertEqual(arr[0].indices.tolist(), [bucket])
        self.assertEqual(arr[0].data.tolist(), [sign])

    def test_same_row_cancels_and_other_bucket_stays(self):
        """Opposite-sign same-bucket hits cancel to a dropped entry; the other bucket stays.

        This is the acceptance scenario: two values in one row hit the same bucket with
        opposite signs, a third hits another bucket. The cancelling bucket must not be
        stored, the third must remain at its original index, and repeated same-sign
        hits in another row must accumulate.
        """
        old, other, separate = _find_signed_collision_pair(16)
        rows = [[old, other, separate], [separate, separate]]
        arr = encoders.HashingEncoder.hash_chunk(
            'md5', np.array(rows, dtype=object), 16, signed=True
        )

        cancelled_bucket = _signed_reference(old, 16)[0]
        kept_bucket, kept_sign = _signed_reference(separate, 16)
        self.assertNotEqual(cancelled_bucket, kept_bucket)

        # Row 0: only the surviving bucket, holding the separate value's single sign.
        self.assertEqual(arr[0].indices.tolist(), [kept_bucket])
        self.assertEqual(arr[0].data.tolist(), [kept_sign])
        # Row 1: two identical same-sign hits accumulate to twice the sign.
        self.assertEqual(arr[1].indices.tolist(), [kept_bucket])
        self.assertEqual(arr[1].data.tolist(), [2 * kept_sign])

    def test_cancelled_bucket_is_not_stored(self):
        """A bucket that cancels to zero occupies no CSR storage."""
        old, other, _ = _find_signed_collision_pair(16)
        arr = encoders.HashingEncoder.hash_chunk(
            'md5', np.array([[old, other]], dtype=object), 16, signed=True
        )
        self.assertEqual(arr.nnz, 0)
        self.assertEqual(arr.data.size, 0)

    def test_no_full_dense_allocation(self):
        """The signed chunk never allocates a rows x buckets dense block."""
        calls = []
        original_zeros = np.zeros

        def spy(shape, *args, **kwargs):
            calls.append(shape)
            return original_zeros(shape, *args, **kwargs)

        np.zeros = spy
        try:
            arr = encoders.HashingEncoder.hash_chunk(
                'md5', np.array([['a', 'b'], ['c', None]], dtype=object), 5000, signed=True
            )
        finally:
            np.zeros = original_zeros

        full = [s for s in calls if isinstance(s, tuple) and len(s) == 2 and s[1] >= 5000]
        self.assertEqual(full, [])
        self.assertEqual(arr.nnz, 3)

    def test_seed_is_reproducible_and_changes_mapping(self):
        """The same seed reproduces the signed output; another seed changes it."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})

        def encode(seed):
            return (
                encoders.HashingEncoder(
                    n_components=16, sparse_output=True, max_process=1, random_state=seed
                )
                .fit_transform(df)
            )

        same_a, same_b, different = encode(7), encode(7), encode(8)
        self.assertEqual((same_a != same_b).nnz, 0)
        self.assertGreater((same_a != different).nnz, 0)

    def test_default_seed_is_fixed(self):
        """Without random_state the signed mapping is still reproducible."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        first = encoders.HashingEncoder(
            n_components=16, sparse_output=True, max_process=1
        ).fit_transform(df)
        second = encoders.HashingEncoder(
            n_components=16, sparse_output=True, max_process=1
        ).fit_transform(df)
        self.assertEqual((first != second).nnz, 0)

    def test_chunking_does_not_change_bucket_values(self):
        """Splitting rows across worker processes leaves every row's buckets unchanged."""
        rng = np.random.RandomState(0)
        df = pd.DataFrame(
            {
                'a': [f'c{rng.randint(0, 40)}' for _ in range(50)],
                'b': [f'd{rng.randint(0, 40)}' for _ in range(50)],
            }
        )
        single = encoders.HashingEncoder(
            n_components=16, sparse_output=True, max_process=1
        ).fit(df).transform(df)
        multi = encoders.HashingEncoder(
            n_components=16, sparse_output=True, max_process=3
        ).fit(df).transform(df)
        self.assertTrue(hasattr(single, 'nnz'))
        self.assertEqual((single != multi).nnz, 0)

    def test_row_reordering_does_not_change_bucket_values(self):
        """Permuting the input rows permutes the output rows and nothing else."""
        rng = np.random.RandomState(1)
        df = pd.DataFrame(
            {
                'a': [f'c{rng.randint(0, 40)}' for _ in range(40)],
                'b': [f'd{rng.randint(0, 40)}' for _ in range(40)],
            }
        )
        encoder = encoders.HashingEncoder(n_components=16, sparse_output=True, max_process=1)
        encoder.fit(df)
        baseline = encoder.transform(df)
        permutation = rng.permutation(len(df))
        shuffled = encoder.transform(df.iloc[permutation].reset_index(drop=True))
        self.assertTrue(np.array_equal(shuffled.toarray(), baseline.toarray()[permutation]))

    def test_transform_returns_signed_csr(self):
        """transform() returns a CSR matrix whose entries carry signs, not counts."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        out = encoders.HashingEncoder(n_components=8, sparse_output=True, max_process=1).fit(
            df
        ).transform(df)
        self.assertTrue(hasattr(out, 'nnz'))
        self.assertEqual(set(np.unique(out.data).tolist()) - {0}, {1, -1})

    def test_sparse_pass_through_columns_appended(self):
        """Non-encoded numeric columns are appended on the right of the sparse matrix."""
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['p', 'q'], 'num': [1.0, 2.0]})
        out = encoders.HashingEncoder(
            n_components=8, sparse_output=True, max_process=1, cols=['a', 'b']
        ).fit(df).transform(df)
        self.assertEqual(out.shape, (2, 9))
        self.assertTrue(np.array_equal(out.toarray()[:, -1], [1.0, 2.0]))

    def test_dense_path_is_unchanged(self):
        """The default dense unsigned output matches the historical md5 counts."""
        df = pd.DataFrame({'a': ['aa', 'bb', 'cc'], 'b': ['x', 'y', 'z']})
        out = encoders.HashingEncoder(n_components=16, max_process=1).fit(df).transform(df)
        expected = np.zeros((3, 16), dtype=int)
        for i, row in enumerate(df.to_numpy()):
            for value in row:
                digest = hashlib.new('md5', str(value).encode('utf-8'))
                expected[i, int.from_bytes(digest.digest(), 'big') % 16] += 1
        self.assertFalse(hasattr(out, 'nnz'))
        self.assertTrue(np.array_equal(out.to_numpy(), expected))

    def test_sparse_output_respects_hash_method(self):
        """The configured hash_method drives the signed mapping too."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        md5_out = encoders.HashingEncoder(
            n_components=8, sparse_output=True, max_process=1, hash_method='md5'
        ).fit_transform(df)
        sha_out = encoders.HashingEncoder(
            n_components=8, sparse_output=True, max_process=1, hash_method='sha256'
        ).fit_transform(df)
        self.assertFalse(np.array_equal(md5_out.toarray(), sha_out.toarray()))

    def test_sparse_drop_invariant_keeps_names_and_shape_in_sync(self):
        """drop_invariant drops zero-variance sparse columns without densifying."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        encoder = encoders.HashingEncoder(
            n_components=64, sparse_output=True, max_process=1, drop_invariant=True
        ).fit(df)
        out = encoder.transform(df)
        self.assertEqual(out.shape[1], len(encoder.get_feature_names_out()))
        self.assertLess(out.shape[1], 64)

    def test_sparse_rejects_non_numeric_pass_through_columns(self):
        """Non-hashed, non-numeric columns cannot live in a sparse matrix."""
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['p', 'q'], 'extra': ['u', 'v']})
        with self.assertRaisesRegex(ValueError, 'non-numeric'):
            encoders.HashingEncoder(
                n_components=8, sparse_output=True, max_process=1, cols=['a', 'b']
            ).fit(df)

    def test_signed_chunk_is_reproducible_via_seed_keyword(self):
        """hash_chunk(..., signed=True, seed=...) is deterministic and seed-sensitive."""
        data = np.array([['x', 'y'], ['z', 'w']], dtype=object)
        first = encoders.HashingEncoder.hash_chunk('md5', data, 16, signed=True, seed=5)
        second = encoders.HashingEncoder.hash_chunk('md5', data, 16, signed=True, seed=5)
        other = encoders.HashingEncoder.hash_chunk('md5', data, 16, signed=True, seed=6)
        self.assertEqual((first != second).nnz, 0)
        self.assertGreater((first != other).nnz, 0)

    def test_sparse_encoder_ignores_global_pandas_transform_config(self):
        """A sparse encoder is not broken by a global transform_output='pandas'."""
        import sklearn

        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['p', 'q']})
        sklearn.set_config(transform_output='pandas')
        try:
            out = encoders.HashingEncoder(
                n_components=8, sparse_output=True, max_process=1
            ).fit_transform(df)
        finally:
            sklearn.set_config(transform_output='default')
        self.assertTrue(hasattr(out, 'nnz'))


