"""Tests for quantile encoder."""
import unittest

import category_encoders as encoders
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError


class TestQuantileEncoder(unittest.TestCase):
    """Tests for percentile encoder."""

    def setUp(self):
        """Create dataframe with categories and a target variable."""
        self.df = pd.DataFrame({'categories': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b']})
        self.target = np.array([1, 2, 0, 4, 5, 0, 6, 7])

    def test_median_works(self):
        """Test that median encoder works.

        Expected output of percentile 50 in df:
            - a median is 4 (a values are 1, 4, 6)
            - b median is 5 (b values are 2, 5, 7)
            - c median is 0 (c values are 0)
        """
        expected_output_median = pd.DataFrame({'categories': [4.0, 5, 0, 4, 5, 0, 4, 5]})

        pd.testing.assert_frame_equal(
            encoders.QuantileEncoder(quantile=0.5, m=0.0).fit_transform(self.df, self.target),
            expected_output_median,
        )

    def test_max_works(self):
        """Test that maximum (=percentile 100) encoder works.

        Expected output of percentile 100 in df:
            - a max is 6
            - b max is 7
            - c max is 0
        """
        expected_output_max = pd.DataFrame({'categories': [6.0, 7, 0, 6, 7, 0, 6, 7]})

        pd.testing.assert_frame_equal(
            encoders.QuantileEncoder(quantile=1, m=0.0).fit_transform(self.df, self.target),
            expected_output_max,
        )

    def test_new_category(self):
        """Test that unknown values are encoded with global mean.

        The global median of the target is 3. If new categories are passed to
        the transformer, then the output should be 3
        """
        transformer_median = encoders.QuantileEncoder(quantile=0.5, m=0.0)
        transformer_median.fit(self.df, self.target)

        new_df = pd.DataFrame({'categories': ['d', 'e']})

        new_medians = pd.DataFrame({'categories': [3.0, 3.0]})

        pd.testing.assert_frame_equal(transformer_median.transform(new_df), new_medians)

    def test_unique_column_collapses_to_prior(self):
        """Test that a fully unique (id-like) column encodes every category to the prior.

        With the default m=1.0, singleton levels used to blend their own target
        quantile 50/50 with the prior, which made id-like columns predictive of
        the label. See issue #327.
        """
        df = pd.DataFrame({'id': [f'row{i}' for i in range(6)]})
        target = np.array([1, 0, 1, 0, 1, 0])

        for quantile in [0.25, 0.5, 0.75]:
            enc = encoders.QuantileEncoder(quantile=quantile)
            result = enc.fit_transform(df, target)
            np.testing.assert_allclose(
                result['id'],
                np.quantile(target, quantile),
                err_msg=f'quantile={quantile}: a unique column must encode to the prior',
            )
            self.assertTrue(all(result.var() < 0.001))

    def test_column_with_repeated_categories_is_not_collapsed(self):
        """Test that the collapse changes outputs only for fully unique columns.

        The fixture mixes repeated levels with a singleton ('d', target 7).
        The column as a whole is not unique, so no level collapses to the
        prior 3.0: 'a', 'b' and 'c' keep their smoothed statistics and the
        singleton 'd' keeps its own smoothed value 5.0. See issue #327.
        """
        df = pd.DataFrame({'categories': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'd']})
        target = np.array([1, 2, 0, 4, 5, 0, 6, 7])
        expected_output = pd.DataFrame(
            {'categories': [3.75, 10 / 3, 1.0, 3.75, 10 / 3, 1.0, 3.75, 5.0]}
        )

        pd.testing.assert_frame_equal(
            encoders.QuantileEncoder(quantile=0.5).fit_transform(df, target),
            expected_output,
        )

    def test_m_still_controls_shrinkage_after_collapse(self):
        """Test that m keeps controlling shrinkage for non-unique columns.

        A large m pulls every level of a repeated-category column close to
        the prior (3.0 for this fixture).
        """
        result = encoders.QuantileEncoder(quantile=0.5, m=1000).fit_transform(self.df, self.target)

        np.testing.assert_allclose(result['categories'], 3.0, atol=0.01)


class TestQuantileEncoderSampleWeight(unittest.TestCase):
    """Frequency weights must reproduce fitting on the expanded data."""

    def test_weighted_quantiles_match_expanded_sample(self):
        """Category and global quantiles equal the expanded six-row sample.

        Targets 0, 10, 20 with weights 1, 4, 1 stand for the six rows
        0, 10, 10, 10, 10, 20. The quantile of category 'a' and the prior used
        for unknown categories must both match numpy's linear interpolation on
        those rows.
        """
        df = pd.DataFrame({'c': ['a', 'a', 'a', 'b']})
        target = np.array([0.0, 10.0, 20.0, 1000.0])
        expanded = np.array([0.0, 10.0, 10.0, 10.0, 10.0, 20.0])

        for quantile in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            enc = encoders.QuantileEncoder(cols=['c'], quantile=quantile, m=0.0)
            enc.fit(df, target, sample_weight=np.array([1, 4, 1, 0]))
            result = enc.transform(pd.DataFrame({'c': ['a', 'unseen']}))

            expected = np.quantile(expanded, quantile)
            np.testing.assert_allclose(result['c'].to_numpy(), [expected, expected])

    def test_zero_weight_extreme_does_not_raise_unknown_fallback(self):
        """A zero-weight extreme target must not lift the prior for unknown categories."""
        df = pd.DataFrame({'c': ['a', 'a', 'a', 'b']})
        target = np.array([0.0, 10.0, 20.0, 1e6])

        weighted = encoders.QuantileEncoder(cols=['c'], quantile=0.75, m=0.0).fit(
            df, target, sample_weight=np.array([1, 4, 1, 0])
        )
        prior = np.quantile([0.0, 10.0, 10.0, 10.0, 10.0, 20.0], 0.75)

        result = weighted.transform(pd.DataFrame({'c': ['never-seen']}))
        self.assertEqual(result['c'].iloc[0], prior)
        self.assertLess(result['c'].iloc[0], 100.0)

    def test_weighted_matches_expanded_fit_end_to_end(self):
        """Weighted fitting equals fitting on the repeated rows, including smoothing."""
        df = pd.DataFrame({'c': ['a', 'a', 'a', 'b']})
        target = np.array([0.0, 10.0, 20.0, 1000.0])
        weights = np.array([1, 4, 1, 0])

        mask = weights > 0
        expanded_df = pd.DataFrame({'c': np.repeat(df['c'].to_numpy()[mask], weights[mask])})
        expanded_target = np.repeat(target[mask], weights[mask])

        unseen = pd.DataFrame({'c': ['a', 'b', 'new']})
        for quantile in [0.1, 0.5, 0.9]:
            for m in [0.0, 1.0, 10.0]:
                weighted = encoders.QuantileEncoder(cols=['c'], quantile=quantile, m=m).fit(
                    df, target, sample_weight=weights
                )
                expanded = encoders.QuantileEncoder(cols=['c'], quantile=quantile, m=m).fit(
                    expanded_df, expanded_target
                )
                np.testing.assert_allclose(
                    weighted.transform(unseen)['c'].to_numpy(),
                    expanded.transform(unseen)['c'].to_numpy(),
                    err_msg=f'quantile={quantile}, m={m}',
                )

    def test_unweighted_output_unchanged(self):
        """Passing explicit unit weights keeps the unweighted result."""
        df = pd.DataFrame({'categories': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b']})
        target = np.array([1, 2, 0, 4, 5, 0, 6, 7])

        for quantile in [0.25, 0.5, 1.0]:
            plain = encoders.QuantileEncoder(quantile=quantile).fit_transform(df, target)
            unit = encoders.QuantileEncoder(quantile=quantile).fit_transform(
                df, target, sample_weight=np.ones(len(target))
            )
            pd.testing.assert_frame_equal(plain, unit)

    def test_weighted_fit_transform_passes_weights(self):
        """fit_transform forwards sample_weight to fit."""
        df = pd.DataFrame({'c': ['a', 'a', 'a']})
        target = np.array([0.0, 10.0, 20.0])
        weights = np.array([1, 4, 1])

        expected = encoders.QuantileEncoder(cols=['c'], quantile=0.5, m=0.0).fit_transform(
            df, target, sample_weight=weights
        )
        np.testing.assert_allclose(expected['c'].to_numpy(), 10.0)

    def test_invalid_sample_weight_is_rejected(self):
        """Weights must be one-dimensional, finite, non-negative and not all zero."""
        df = pd.DataFrame({'c': ['a', 'a', 'b']})
        target = np.array([0.0, 1.0, 2.0])
        encoder = encoders.QuantileEncoder(cols=['c'])

        with self.assertRaises(ValueError):
            encoder.fit(df, target, sample_weight=np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            encoder.fit(df, target, sample_weight=np.array([1.0, -1.0, 1.0]))
        with self.assertRaises(ValueError):
            encoder.fit(df, target, sample_weight=np.array([0.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):
            encoder.fit(df, target, sample_weight=np.array([1.0, np.inf, 1.0]))

    def test_category_with_only_zero_weight_rows_falls_back_to_prior(self):
        """A category seen only on zero-weight rows is known but encodes as the prior."""
        df = pd.DataFrame({'c': ['a', 'a', 'z']})
        target = np.array([0.0, 10.0, 1e6])

        encoder = encoders.QuantileEncoder(cols=['c'], quantile=0.5, m=0.0).fit(
            df, target, sample_weight=np.array([1.0, 1.0, 0.0])
        )
        result = encoder.transform(pd.DataFrame({'c': ['a', 'z']}))
        self.assertEqual(result['c'].iloc[0], 5.0)
        self.assertEqual(result['c'].iloc[1], 5.0)

    def test_id_like_detection_follows_expanded_rows(self):
        """A weighted column collapses to the prior only if the expanded rows are unique.

        Five distinct categories with unit weights are id-like and collapse to
        the prior. Repeating one of them makes the expanded sample non-unique,
        so no category collapses any more.
        """
        df = pd.DataFrame({'id': [f'row{i}' for i in range(5)]})
        target = np.array([0.0, 1.0, 0.0, 1.0, 9.0])

        unit = encoders.QuantileEncoder(cols=['id'], quantile=0.5).fit(
            df, target, sample_weight=np.ones(5)
        )
        np.testing.assert_allclose(unit.transform(df)['id'].to_numpy(), np.quantile(target, 0.5))

        repeated = encoders.QuantileEncoder(cols=['id'], quantile=0.5).fit(
            df, target, sample_weight=np.array([1, 1, 2, 1, 1])
        )
        expanded_target = np.repeat(target, [1, 1, 2, 1, 1])
        expanded = encoders.QuantileEncoder(cols=['id'], quantile=0.5).fit(
            pd.DataFrame({'id': np.repeat(df['id'].to_numpy(), [1, 1, 2, 1, 1])}),
            expanded_target,
        )
        np.testing.assert_allclose(
            repeated.transform(df)['id'].to_numpy(), expanded.transform(df)['id'].to_numpy()
        )
        self.assertGreater(repeated.transform(df)['id'].nunique(), 1)

    def test_series_weights_are_taken_positionally(self):
        """A pandas Series of weights is not aligned on its index."""
        df = pd.DataFrame({'c': ['a', 'a', 'a']})
        target = np.array([0.0, 10.0, 20.0])
        weights = pd.Series([1, 4, 1], index=[7, 8, 9])

        encoder = encoders.QuantileEncoder(cols=['c'], quantile=0.5, m=0.0).fit(
            df, target, sample_weight=weights
        )
        self.assertEqual(encoder.transform(pd.DataFrame({'c': ['a']}))['c'].iloc[0], 10.0)


class TestSummaryEncoder(unittest.TestCase):
    """Tests for summary encoder."""

    def setUp(self):
        """Create dataframe with categories and a target variable."""
        self.df = pd.DataFrame({'categories': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b']})
        self.target = np.array([1, 2, 0, 4, 5, 0, 6, 7])
        self.col = 'categories'

    def assert_same_quantile(self, quantile):
        """Check the summary encoder with a single quantile coincides with the quantile encoder."""
        quantile_results = encoders.QuantileEncoder(
            cols=[self.col], quantile=quantile
        ).fit_transform(self.df, self.target)

        summary_results = encoders.SummaryEncoder(
            cols=[self.col], quantiles=[quantile]
        ).fit_transform(self.df, self.target)

        percentile = round(quantile * 100)
        col_name = str(self.col) + '_' + str(percentile)
        np.testing.assert_allclose(
            quantile_results[self.col].values,
            summary_results[col_name].values,
        )

    def test_several_quantiles(self):
        """Check that all quantiles of the QE are in the summary encoder."""
        for quantile in [0.1, 0.5, 0.9]:
            self.assert_same_quantile(quantile)

    def test_several_quantiles_reverse(self):
        """Check that all quantiles of summary encoder are in the quantile encoder."""
        quantile_list = [0.2, 0.1, 0.8]

        summary_results = encoders.SummaryEncoder(
            cols=[self.col], quantiles=quantile_list
        ).fit_transform(self.df, self.target)

        for quantile in quantile_list:
            quantile_results = encoders.QuantileEncoder(
                cols=[self.col], quantile=quantile
            ).fit_transform(self.df, self.target)

            percentile = round(quantile * 100)
            col_name = str(self.col) + '_' + str(percentile)

            np.testing.assert_allclose(
                quantile_results[self.col].values,
                summary_results[col_name].values,
            )

    def test_sample_weight_is_forwarded_to_every_quantile(self):
        """SummaryEncoder passes frequency weights on to each QuantileEncoder."""
        df = pd.DataFrame({'c': ['a', 'a', 'a', 'b']})
        target = np.array([0.0, 10.0, 20.0, 1000.0])
        weights = np.array([1, 4, 1, 0])
        unseen = pd.DataFrame({'c': ['a', 'new']})

        summary = encoders.SummaryEncoder(cols=['c'], quantiles=[0.25, 0.5, 0.75], m=0.0).fit(
            df, target, sample_weight=weights
        )
        summary_result = summary.transform(unseen)
        for quantile in [0.25, 0.5, 0.75]:
            quantile_result = encoders.QuantileEncoder(
                cols=['c'], quantile=quantile, m=0.0
            ).fit(df, target, sample_weight=weights)
            np.testing.assert_allclose(
                summary_result[f'c_{round(quantile * 100)}'].to_numpy(),
                quantile_result.transform(unseen)['c'].to_numpy(),
            )


class TestSummaryEncoderHardening(unittest.TestCase):
    """Hardening tests for SummaryEncoder mutation survivors."""

    def setUp(self):
        """Shared six-row fixture with three categories and a float target."""
        self.df = pd.DataFrame({'c': ['a', 'a', 'b', 'b', 'c', 'c']})
        self.target = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_fit_transform_matches_fit_then_transform_with_target(self):
        """fit_transform must pass the target through to transform.

        Mutant in category_encoders/quantile_encoder.py::SummaryEncoder
        fit_transform (transform(X, None)) drops y, silently producing
        unsupervised output instead of the fit-then-transform result.
        """
        fit_transform_out = encoders.SummaryEncoder(quantiles=[0.25, 0.75]).fit_transform(
            self.df, self.target
        )
        encoder = encoders.SummaryEncoder(quantiles=[0.25, 0.75])
        encoder.fit(self.df, self.target)
        transform_out = encoder.transform(self.df, self.target)
        pd.testing.assert_frame_equal(fit_transform_out, transform_out)

    def test_get_feature_names_in_requires_fit(self):
        """get_feature_names_in before fit raises NotFittedError.

        Mutant dropping the getattr default in get_feature_names_in turns
        the documented NotFittedError into a bare AttributeError.
        """
        encoder = encoders.SummaryEncoder(quantiles=[0.25, 0.75])
        with self.assertRaises(NotFittedError):
            encoder.get_feature_names_in()

    def test_feature_names_out_is_object_array(self):
        """Feature names come back as an object-dtype numpy array.

        Mutant np.array(out_feats, dtype=object) -> dtype=None returns a
        '<U...' unicode array instead of object dtype.
        """
        encoder = encoders.SummaryEncoder(quantiles=[0.25, 0.75]).fit(self.df, self.target)
        names = encoder.get_feature_names_out()
        self.assertIsInstance(names, np.ndarray)
        self.assertEqual(names.dtype, object)

