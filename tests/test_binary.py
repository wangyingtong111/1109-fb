"""Tests for the BinaryEncoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd

from category_encoders.basen import AMBIGUOUS, NO_MATCH, RECOVERED


class TestBinaryEncoder(TestCase):
    """Unit tests for the BinaryEncoder."""

    def test_binary_bin(self):
        """Test the BinaryEncoder with only two values."""
        data = np.array(['a', 'ba', 'ba'])
        out = encoders.BinaryEncoder().fit_transform(data)
        self.assertTrue(pd.DataFrame([[0, 1], [1, 0], [1, 0]], columns=['0_0', '0_1']).equals(out))

    def test_binary_dist(self):
        """Test the BinaryEncoder with all distinct values."""
        data = np.array(['apple', 'orange', 'peach', 'lemon'])
        encoder = encoders.BinaryEncoder()
        encoder.fit(data)

        # split dataframe into two transforms and recombine
        a = encoder.transform(data[:1])
        b = encoder.transform(data[1:])
        split = pd.concat([a, b])
        split = split.reset_index(drop=True)

        # run all at once
        c = encoder.transform(data)

        # make sure they are the same
        self.assertTrue(split.equals(c))


class TestBinaryPartialInverse(TestCase):
    """Tests for the partial (erased-bit) inverse of the BinaryEncoder."""

    def setUp(self):
        """Four categories map to 001, 010, 011, 100."""
        self.fit_df = pd.DataFrame({'a': ['w', 'x', 'y', 'z']})
        self.encoder = encoders.BinaryEncoder(cols=['a']).fit(self.fit_df)
        expected_words = pd.DataFrame(
            [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]],
            index=[1, 2, 3, 4],
            columns=['a_0', 'a_1', 'a_2'],
        )
        pd.testing.assert_frame_equal(
            expected_words, self.encoder.mapping[0]['mapping'].loc[1:4]
        )

    def test_erased_bit_unique_and_ambiguous_recovery(self):
        """10? only matches 100, while 01? matches both 010 and 011."""
        batch = pd.DataFrame({
            'a_0': [1, 0, 0, 0],
            'a_1': [0, 1, 0, 0],
            'a_2': [np.nan, np.nan, 1, 1],
        })
        recovered, status = self.encoder.inverse_transform_partial(batch)

        self.assertListEqual(['a'], recovered.columns.tolist())
        self.assertListEqual(['a_recovery_status'], status.columns.tolist())
        # 10? -> 100 (z); 01? -> 010/011 ambiguous; 001 -> w; ?01 -> 001 (w)
        self.assertEqual('z', recovered['a'].iloc[0])
        self.assertTrue(pd.isna(recovered['a'].iloc[1]))
        self.assertEqual('w', recovered['a'].iloc[2])
        self.assertEqual('w', recovered['a'].iloc[3])
        self.assertListEqual(
            [RECOVERED, AMBIGUOUS, RECOVERED, RECOVERED],
            status['a_recovery_status'].tolist(),
        )

    def test_reserved_all_zero_word_is_not_a_category_candidate(self):
        """The all-zero unknown/missing word must never recover a normal category."""
        reserved = pd.DataFrame({'a_0': [0], 'a_1': [0], 'a_2': [0]})
        recovered, status = self.encoder.inverse_transform_partial(reserved)
        self.assertTrue(pd.isna(recovered['a'].iloc[0]))
        self.assertEqual(NO_MATCH, status['a_recovery_status'].iloc[0])

    def test_invalid_bit_value_is_no_match(self):
        """A digit outside {0, 1} cannot match any binary code word."""
        invalid = pd.DataFrame({'a_0': [7], 'a_1': [0], 'a_2': [1]})
        recovered, status = self.encoder.inverse_transform_partial(invalid)
        self.assertTrue(pd.isna(recovered['a'].iloc[0]))
        self.assertEqual(NO_MATCH, status['a_recovery_status'].iloc[0])

    def test_fully_erased_word_is_ambiguous_for_multiple_categories(self):
        """With every bit erased, all fitted categories are candidates."""
        erased = pd.DataFrame({
            'a_0': [np.nan], 'a_1': [np.nan], 'a_2': [np.nan]
        })
        recovered, status = self.encoder.inverse_transform_partial(erased)
        self.assertTrue(pd.isna(recovered['a'].iloc[0]))
        self.assertEqual(AMBIGUOUS, status['a_recovery_status'].iloc[0])

    def test_complete_words_round_trip_as_recovered(self):
        """Partial inverse on clean output reproduces the exact inverse."""
        encoded = self.encoder.transform(self.fit_df)
        recovered, status = self.encoder.inverse_transform_partial(encoded)
        self.assertListEqual(['w', 'x', 'y', 'z'], recovered['a'].tolist())
        self.assertTrue((status['a_recovery_status'] == RECOVERED).all())

    def test_inverse_transform_partial_flag_and_alias(self):
        """The partial mode is reachable through the compatible flag and alias."""
        batch = pd.DataFrame({
            'a_0': [1, 0], 'a_1': [0, 1], 'a_2': [np.nan, np.nan]
        })
        recovered, status = self.encoder.inverse_transform(batch, partial=True)
        alias_recovered, alias_status = self.encoder.partial_inverse_transform(batch)
        pd.testing.assert_frame_equal(recovered, alias_recovered)
        pd.testing.assert_frame_equal(status, alias_status)

    def test_invariant_bit_cleared_downstream_is_reconstructed(self):
        """A single-category column whose constant bit was dropped still recovers."""
        fit_df = pd.DataFrame({'c': ['only', 'only', 'only']})
        encoder = encoders.BinaryEncoder(cols=['c'], drop_invariant=True).fit(fit_df)
        self.assertEqual(['c_0'], encoder.invariant_cols)
        downstream = encoder.transform(fit_df)
        self.assertEqual(0, downstream.shape[1])

        recovered, status = encoder.inverse_transform_partial(downstream)
        self.assertListEqual(['only', 'only', 'only'], recovered['c'].tolist())
        self.assertTrue((status['c_recovery_status'] == RECOVERED).all())

        # the standard exact interface keeps its strict drop_invariant guard
        with self.assertRaises(ValueError):
            encoder.inverse_transform(downstream)

    def test_compound_two_fields_passthrough_reorder_and_duplicate_index(self):
        """The prompt's compound case is handled row-wise and field-wise."""
        fit_df = pd.DataFrame({
            'fixed': ['alpha'] * 4,
            'flaky': ['a', 'b', 'c', 'd'],
            'ordinary': [10, 20, 30, 40],
        })
        encoder = encoders.BinaryEncoder(
            cols=['fixed', 'flaky'], drop_invariant=True
        ).fit(fit_df)
        self.assertEqual(['fixed_0'], encoder.invariant_cols)

        # fixed_0 is absent because it is invariant. Reorder all remaining
        # columns and repeat index labels.
        batch = pd.DataFrame({
            'flaky_1': [0, 1, 1, 0],
            'ordinary': [10, 20, 30, 40],
            'flaky_0': [1, 0, 0, 0],
            'flaky_2': [np.nan, np.nan, 1, 0],
        }, index=[7, 7, 9, 9])

        recovered, status = encoder.inverse_transform(batch, partial=True)

        self.assertListEqual([7, 7, 9, 9], recovered.index.tolist())
        self.assertListEqual([7, 7, 9, 9], status.index.tolist())
        self.assertListEqual(['alpha'] * 4, recovered['fixed'].tolist())
        self.assertEqual('d', recovered['flaky'].iloc[0])  # 10?
        self.assertTrue(pd.isna(recovered['flaky'].iloc[1]))  # 01?
        self.assertEqual('c', recovered['flaky'].iloc[2])
        self.assertTrue(pd.isna(recovered['flaky'].iloc[3]))  # 000 reserved
        self.assertListEqual([10, 20, 30, 40], recovered['ordinary'].tolist())
        self.assertListEqual(
            [RECOVERED, AMBIGUOUS, RECOVERED, NO_MATCH],
            status['flaky_recovery_status'].tolist(),
        )
        self.assertTrue(status['fixed_recovery_status'].all())

    def test_missing_bit_column_is_treated_as_a_fully_erased_bit(self):
        """An absent column supplies no known digit but does not disable matching."""
        # Equivalent to 10? -> z and 01? -> ambiguous.
        batch = pd.DataFrame({'a_0': [1, 0], 'a_1': [0, 1]})
        recovered, status = self.encoder.inverse_transform_partial(batch)
        self.assertEqual('z', recovered['a'].iloc[0])
        self.assertTrue(pd.isna(recovered['a'].iloc[1]))
        self.assertEqual(RECOVERED, status['a_recovery_status'].iloc[0])
        self.assertEqual(AMBIGUOUS, status['a_recovery_status'].iloc[1])

    def test_partial_inverse_does_not_mutate_input(self):
        """Erased-bit decoding must leave the caller's frame untouched."""
        batch = pd.DataFrame({
            'a_0': [1, 0], 'a_1': [0, 1], 'a_2': [np.nan, np.nan]
        })
        snapshot = batch.copy(deep=True)
        self.encoder.inverse_transform_partial(batch)
        pd.testing.assert_frame_equal(snapshot, batch)

    def test_repeated_erasure_masks_are_resolved_consistently(self):
        """Identical masks across many rows yield identical per-row outcomes."""
        rng = np.random.default_rng(42)
        words = self.encoder.mapping[0]['mapping'].loc[1:4].to_numpy()
        n = 500
        big = words[rng.integers(0, 4, size=n)].astype(float).copy()
        big[rng.random((n, 3)) < 0.4] = np.nan
        big_df = pd.DataFrame(big, columns=['a_0', 'a_1', 'a_2'])

        recovered, status = self.encoder.inverse_transform_partial(big_df)

        vocabulary = words.astype(float)
        labels = np.array([1, 2, 3, 4])
        expected_status, expected_label = [], []
        for row in big:
            erased = np.isnan(row)
            matches = [
                label
                for label, word in zip(labels, vocabulary)
                if np.array_equal(word[~erased], row[~erased])
            ]
            if len(matches) == 1:
                expected_status.append(RECOVERED)
                expected_label.append(matches[0])
            elif len(matches) > 1:
                expected_status.append(AMBIGUOUS)
                expected_label.append(np.nan)
            else:
                expected_status.append(NO_MATCH)
                expected_label.append(np.nan)
        self.assertListEqual(expected_status, status['a_recovery_status'].tolist())

        ordinal = self.encoder.ordinal_encoder.mapping[0]['mapping']
        category_ordinal = ordinal[ordinal > 0]
        got_ordinal = recovered['a'].map(
            pd.Series(category_ordinal.array, index=category_ordinal.index)
        ).to_numpy(float)
        expected_ordinal = np.array(expected_label, dtype=float)
        np.testing.assert_array_equal(np.isnan(got_ordinal), np.isnan(expected_ordinal))
        np.testing.assert_array_equal(
            got_ordinal[~np.isnan(got_ordinal)],
            expected_ordinal[~np.isnan(expected_ordinal)],
        )
