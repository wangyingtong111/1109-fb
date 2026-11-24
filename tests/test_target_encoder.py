"""Tests for the TargetEncoder class."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd
from category_encoders.datasets import load_compass, load_postcodes
from sklearn.compose import ColumnTransformer

import tests.helpers as th


class TestTargetEncoder(TestCase):
    """Unit tests for the Target Encoder."""

    def setUp(self):
        """Set up the test case."""
        self.hierarchical_cat_example = pd.DataFrame(
            {
                'Compass': [
                    'N',
                    'N',
                    'NE',
                    'NE',
                    'NE',
                    'SE',
                    'SE',
                    'S',
                    'S',
                    'S',
                    'S',
                    'W',
                    'W',
                    'W',
                    'W',
                    'W',
                ],
                'Speed': [
                    'slow',
                    'slow',
                    'slow',
                    'slow',
                    'medium',
                    'medium',
                    'medium',
                    'fast',
                    'fast',
                    'fast',
                    'fast',
                    'fast',
                    'fast',
                    'fast',
                    'fast',
                    'fast',
                ],
                'Animal': [
                    'Cat',
                    'Cat',
                    'Cat',
                    'Cat',
                    'Cat',
                    'Dog',
                    'Dog',
                    'Dog',
                    'Dog',
                    'Dog',
                    'Dog',
                    'Tiger',
                    'Tiger',
                    'Wolf',
                    'Wolf',
                    'Cougar',
                ],
                'Plant': [
                    'Rose',
                    'Rose',
                    'Rose',
                    'Rose',
                    'Daisy',
                    'Daisy',
                    'Daisy',
                    'Daisy',
                    'Daffodil',
                    'Daffodil',
                    'Daffodil',
                    'Daffodil',
                    'Bluebell',
                    'Bluebell',
                    'Bluebell',
                    'Bluebell',
                ],
                'target': [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            },
            columns=['Compass', 'Speed', 'Animal', 'Plant', 'target'],
        )
        self.hierarchical_map = {
            'Compass': {'N': ('N', 'NE'), 'S': ('S', 'SE'), 'W': 'W'},
            'Animal': {'Feline': ('Cat', 'Tiger', 'Cougar'), 'Canine': ('Dog', 'Wolf')},
            'Plant': {
                'Flower': ('Rose', 'Daisy', 'Daffodil', 'Bluebell'),
                'Tree': ('Ash', 'Birch'),
            },
        }

    def test_target_encoder(self):
        """Test that no error occurs when calling the target encoder."""
        np_X = th.create_array(n_rows=100)
        np_X_t = th.create_array(n_rows=50, extras=True)
        np_y = np.random.randn(np_X.shape[0]) > 0.5
        np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
        X = th.create_dataset(n_rows=100)
        X_t = th.create_dataset(n_rows=50, extras=True)
        y = pd.DataFrame(np_y)
        y_t = pd.DataFrame(np_y_t)
        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2)
        enc.fit(X, y)
        th.verify_numeric(enc.transform(X_t))
        th.verify_numeric(enc.transform(X_t, y_t))

    def test_fit(self):
        """Test the fit method and correct values are fitted."""
        k = 2
        f = 10
        binary_cat_example = pd.DataFrame(
            {
                'Trend': ['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT'],
                'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            }
        )
        encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
        encoder.fit(binary_cat_example, binary_cat_example['target'])
        trend_mapping = encoder.mapping['Trend']
        ordinal_mapping = encoder.ordinal_encoder.category_mapping[0]['mapping']

        self.assertAlmostEqual(0.4125, trend_mapping[ordinal_mapping.loc['DOWN']], delta=1e-4)
        self.assertEqual(0.5, trend_mapping[ordinal_mapping.loc['FLAT']])
        self.assertAlmostEqual(0.5874, trend_mapping[ordinal_mapping.loc['UP']], delta=1e-4)


    def test_fit_transform(self):
        """Test the good case without unknowns or NaN values."""
        k = 2
        f = 10
        training_data = ['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT']
        cases = {"with list input": training_data,
                 "with pd categorical input": pd.Categorical(training_data,
                                                             categories=['UP', 'FLAT', 'DOWN'])}
        target = [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]

        for case, input_data in cases.items():
            with self.subTest(case):
                binary_cat_example = pd.DataFrame(
                    {
                        'Trend': input_data,
                        'target': target
                    }
                )
                encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
                result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
                values = result['Trend'].array
                self.assertAlmostEqual(0.5874, values[0], delta=1e-4)
                self.assertAlmostEqual(0.5874, values[1], delta=1e-4)
                self.assertAlmostEqual(0.4125, values[2], delta=1e-4)
                self.assertEqual(0.5, values[3])

    def test_fit_transform_with_nan(self):
        """Test that the encoder works with NaN values."""
        k = 2
        f = 10
        binary_cat_example = pd.DataFrame(
            {
                'Trend': pd.Series(
                    [np.nan, np.nan, 'DOWN', 'FLAT', 'DOWN', np.nan, 'DOWN', 'FLAT', 'FLAT', 'FLAT']
                ),
                'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            }
        )
        encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
        result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
        values = result['Trend'].array
        self.assertAlmostEqual(0.5874, values[0], delta=1e-4)
        self.assertAlmostEqual(0.5874, values[1], delta=1e-4)
        self.assertAlmostEqual(0.4125, values[2], delta=1e-4)
        self.assertEqual(0.5, values[3])

    def test_handle_missing_value(self):
        """Should set the global mean value for missing values if handle_missing=value."""
        df = pd.DataFrame(
            {'color': ['a', 'a', 'a', 'b', 'b', 'b'], 'outcome': [1.6, 0, 0, 1, 0, 1]}
        )

        train = df.drop('outcome', axis=1)
        target = df.drop('color', axis=1)
        test = pd.Series([np.nan, 'b'], name='color')
        test_target = pd.Series([0, 0])

        enc = encoders.TargetEncoder(cols=['color'], handle_missing='value')
        enc.fit(train, target['outcome'])
        obtained = enc.transform(test, test_target)

        self.assertEqual(0.6, list(obtained['color'])[0])

    def test_handle_unknown_value(self):
        """Test that encoder sets the global mean value for unknown values."""
        train = pd.Series(['a', 'a', 'a', 'b', 'b', 'b'], name='color')
        target = pd.Series([1.6, 0, 0, 1, 0, 1], name='target')
        test = pd.Series(['c', 'b'], name='color')
        test_target = pd.Series([0, 0])

        enc = encoders.TargetEncoder(cols=['color'], handle_unknown='value')
        enc.fit(train, target)
        obtained = enc.transform(test, test_target)

        self.assertEqual(0.6, list(obtained['color'])[0])

    def test_hierarchical_smoothing(self):
        """Test that encoder works with a hierarchical mapping."""
        enc = encoders.TargetEncoder(
            verbose=1,
            smoothing=2,
            min_samples_leaf=2,
            hierarchy=self.hierarchical_map,
            cols=['Compass'],
        )
        result = enc.fit_transform(
            self.hierarchical_cat_example, self.hierarchical_cat_example['target']
        )
        values = result['Compass'].array
        self.assertAlmostEqual(0.6226, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9038, values[2], delta=1e-4)
        self.assertAlmostEqual(0.1766, values[5], delta=1e-4)
        self.assertAlmostEqual(0.4605, values[7], delta=1e-4)
        self.assertAlmostEqual(0.4033, values[11], delta=1e-4)

    def test_hierarchical_smoothing_multi(self):
        """Test that the encoder works with multiple columns."""
        enc = encoders.TargetEncoder(
            verbose=1,
            smoothing=2,
            min_samples_leaf=2,
            hierarchy=self.hierarchical_map,
            cols=['Compass', 'Speed', 'Animal'],
        )
        result = enc.fit_transform(
            self.hierarchical_cat_example, self.hierarchical_cat_example['target']
        )

        values = result['Compass'].array
        self.assertAlmostEqual(0.6226, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9038, values[2], delta=1e-4)
        self.assertAlmostEqual(0.1766, values[5], delta=1e-4)
        self.assertAlmostEqual(0.4605, values[7], delta=1e-4)
        self.assertAlmostEqual(0.4033, values[11], delta=1e-4)

        values = result['Speed'].array
        self.assertAlmostEqual(0.6827, values[0], delta=1e-4)
        self.assertAlmostEqual(0.3962, values[4], delta=1e-4)
        self.assertAlmostEqual(0.4460, values[7], delta=1e-4)

        values = result['Animal'].array
        self.assertAlmostEqual(0.7887, values[0], delta=1e-4)
        self.assertAlmostEqual(0.3248, values[5], delta=1e-4)
        self.assertAlmostEqual(0.6190, values[11], delta=1e-4)
        self.assertAlmostEqual(0.1309, values[13], delta=1e-4)
        self.assertAlmostEqual(0.8370, values[15], delta=1e-4)

    def test_hierarchical_part_named_cols(self):
        """Test that the encoder works with a partial hierarchy."""
        enc = encoders.TargetEncoder(
            verbose=1,
            smoothing=2,
            min_samples_leaf=2,
            hierarchy=self.hierarchical_map,
            cols=['Compass'],
        )
        result = enc.fit_transform(
            self.hierarchical_cat_example, self.hierarchical_cat_example['target']
        )

        values = result['Compass'].array
        self.assertAlmostEqual(0.6226, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9038, values[2], delta=1e-4)
        self.assertAlmostEqual(0.1766, values[5], delta=1e-4)
        self.assertAlmostEqual(0.4605, values[7], delta=1e-4)
        self.assertAlmostEqual(0.4033, values[11], delta=1e-4)

        values = result['Speed'].array
        self.assertEqual('slow', values[0])

    def test_hierarchy_pandas_index(self):
        """Test that the encoder works with a pandas index."""
        df = pd.DataFrame(
            {
                'hello': ['a', 'b', 'c', 'a', 'a', 'b', 'c', 'd', 'd'],
                'world': [0, 1, 0, 0, 1, 0, 0, 1, 1],
            },
            columns=pd.Index(['hello', 'world']),
        )
        cols = df.select_dtypes(include='object').columns

        self.hierarchical_map = {
            'hello': {'A': ('a', 'b'), 'B': ('c', 'd')},
        }

        enc = encoders.TargetEncoder(
            verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=self.hierarchical_map, cols=cols
        )
        result = enc.fit_transform(df, df['world'])

        values = result['hello'].array
        self.assertAlmostEqual(0.3616, values[0], delta=1e-4)
        self.assertAlmostEqual(0.4541, values[1], delta=1e-4)
        self.assertAlmostEqual(0.2425, values[2], delta=1e-4)
        self.assertAlmostEqual(0.7425, values[7], delta=1e-4)

    def test_hierarchy_single_mapping(self):
        """Test that mapping a single column works."""
        enc = encoders.TargetEncoder(
            verbose=1,
            smoothing=2,
            min_samples_leaf=2,
            hierarchy=self.hierarchical_map,
            cols=['Plant'],
        )
        result = enc.fit_transform(
            self.hierarchical_cat_example, self.hierarchical_cat_example['target']
        )

        values = result['Plant'].array
        self.assertAlmostEqual(0.6828, values[0], delta=1e-4)
        self.assertAlmostEqual(0.5, values[4], delta=1e-4)
        self.assertAlmostEqual(0.5, values[8], delta=1e-4)
        self.assertAlmostEqual(0.3172, values[12], delta=1e-4)

    def test_hierarchy_no_mapping(self):
        """Test that a trivial hierarchy mapping label to itself works."""
        hierarchical_map = {
            'Plant': {
                'Rose': 'Rose',
                'Daisy': 'Daisy',
                'Daffodil': 'Daffodil',
                'Bluebell': 'Bluebell',
            }
        }

        enc = encoders.TargetEncoder(
            verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map, cols=['Plant']
        )
        result = enc.fit_transform(
            self.hierarchical_cat_example, self.hierarchical_cat_example['target']
        )

        values = result['Plant'].array
        self.assertAlmostEqual(0.6828, values[0], delta=1e-4)
        self.assertAlmostEqual(0.5, values[4], delta=1e-4)
        self.assertAlmostEqual(0.5, values[8], delta=1e-4)
        self.assertAlmostEqual(0.3172, values[12], delta=1e-4)

    def test_hierarchy_error(self):
        """Test that an error is raised when the hierarchy dictionary is invalid."""
        hierarchical_map = {'Plant': {'Flower': {'Rose': ('Pink', 'Yellow', 'Red')}, 'Tree': 'Ash'}}
        with self.assertRaises(ValueError):
            encoders.TargetEncoder(
                verbose=1,
                smoothing=2,
                min_samples_leaf=2,
                hierarchy=hierarchical_map,
                cols=['Plant'],
            )

    def test_trivial_hierarchy(self):
        """Test that a trivial hierarchy works."""
        trivial_hierarchical_map = {'Plant': {'Plant': ('Rose', 'Daisy', 'Daffodil', 'Bluebell')}}

        enc_hier = encoders.TargetEncoder(
            verbose=1,
            smoothing=2,
            min_samples_leaf=2,
            hierarchy=trivial_hierarchical_map,
            cols=['Plant'],
        )
        result_hier = enc_hier.fit_transform(
            self.hierarchical_cat_example, self.hierarchical_cat_example['target']
        )
        enc_no_hier = encoders.TargetEncoder(
            verbose=1, smoothing=2, min_samples_leaf=2, cols=['Plant']
        )
        result_no_hier = enc_no_hier.fit_transform(
            self.hierarchical_cat_example, self.hierarchical_cat_example['target']
        )
        pd.testing.assert_series_equal(result_hier['Plant'], result_no_hier['Plant'])

    def test_hierarchy_multi_level(self):
        """Test that hierarchy works with a multi-level hierarchy."""
        hierarchy_multi_level_df = pd.DataFrame(
            {
                'Animal': [
                    'Cat',
                    'Cat',
                    'Dog',
                    'Dog',
                    'Dog',
                    'Osprey',
                    'Kite',
                    'Kite',
                    'Carp',
                    'Carp',
                    'Carp',
                    'Clownfish',
                    'Clownfish',
                    'Lizard',
                    'Snake',
                    'Snake',
                ],
                'target': [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            },
            columns=['Animal', 'target'],
        )
        hierarchy_multi_level = {
            'Animal': {
                'Warm-Blooded': {
                    'Mammals': ('Cat', 'Dog'),
                    'Birds': ('Osprey', 'Kite'),
                    'Fish': ('Carp', 'Clownfish'),
                },
                'Cold-Blooded': {'Reptiles': ('Lizard'), 'Amphibians': ('Snake', 'Frog')},
            }
        }

        enc = encoders.TargetEncoder(
            verbose=1,
            smoothing=2,
            min_samples_leaf=2,
            hierarchy=hierarchy_multi_level,
            cols=['Animal'],
        )
        result = enc.fit_transform(hierarchy_multi_level_df, hierarchy_multi_level_df['target'])

        values = result['Animal'].array
        self.assertAlmostEqual(0.6261, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9065, values[2], delta=1e-4)
        self.assertAlmostEqual(0.2556, values[5], delta=1e-4)
        self.assertAlmostEqual(0.3680, values[8], delta=1e-4)
        self.assertAlmostEqual(0.4626, values[11], delta=1e-4)
        self.assertAlmostEqual(0.1535, values[13], delta=1e-4)
        self.assertAlmostEqual(0.4741, values[14], delta=1e-4)

    def test_hierarchy_columnwise_compass(self):
        """Test that hierarchy works with a columnwise hierarchy."""
        X, y = load_compass()
        cols = X.columns[~X.columns.str.startswith('HIER')]
        HIER_cols = X.columns[X.columns.str.startswith('HIER')]
        enc = encoders.TargetEncoder(
            verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=X[HIER_cols], cols=['compass']
        )
        result = enc.fit_transform(X[cols], y)

        values = result['compass'].array
        self.assertAlmostEqual(0.6226, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9038, values[2], delta=1e-4)
        self.assertAlmostEqual(0.1766, values[5], delta=1e-4)
        self.assertAlmostEqual(0.4605, values[7], delta=1e-4)
        self.assertAlmostEqual(0.4033, values[11], delta=1e-4)

    def test_hierarchy_columnwise_postcodes(self):
        """Test that hierarchy works with a columnwise hierarchy."""
        X, y = load_postcodes('binary')
        cols = X.columns[~X.columns.str.startswith('HIER')]
        HIER_cols = X.columns[X.columns.str.startswith('HIER')]
        enc = encoders.TargetEncoder(
            verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=X[HIER_cols], cols=['postcode']
        )
        result = enc.fit_transform(X[cols], y)

        values = result['postcode'].array
        self.assertAlmostEqual(0.8448, values[0], delta=1e-4)

    def test_hierarchy_columnwise_missing_level(self):
        """Test that an error is raised when a hierarchy is given but a sub-column is missing."""
        X, y = load_postcodes('binary')
        HIER_cols = ['HIER_postcode_1', 'HIER_postcode_2', 'HIER_postcode_4']
        with self.assertRaises(ValueError):
            encoders.TargetEncoder(
                verbose=1,
                smoothing=2,
                min_samples_leaf=2,
                hierarchy=X[HIER_cols],
                cols=['postcode'],
            )

    def test_hierarchy_mapping_no_cols(self):
        """Test that an error is raised when the hierarchy is given but no columns to encode."""
        hierarchical_map = {'Compass': {'N': ('N', 'NE'), 'S': ('S', 'SE'), 'W': 'W'}}
        with self.assertRaises(ValueError):
            encoders.TargetEncoder(
                verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map
            )

    def test_hierarchy_mapping_cols_missing(self):
        """Test that an error is raised when the dataframe is missing the hierarchy column."""
        X = ['N', 'N', 'NE', 'NE', 'NE', 'SE', 'SE', 'S', 'S', 'S', 'S', 'W', 'W', 'W', 'W', 'W']
        hierarchical_map = {'Compass': {'N': ('N', 'NE'), 'S': ('S', 'SE'), 'W': 'W'}}
        y = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        enc = encoders.TargetEncoder(
            verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map, cols=['Compass']
        )
        with self.assertRaises(ValueError):
            enc.fit_transform(X, y)

    def _fit_store_region_encoder(self):
        """Fit an encoder whose North/South regions have clearly different means."""
        train = pd.DataFrame(
            {
                'store': ['N1', 'N1', 'N1', 'N2', 'N2', 'S1', 'S1', 'S1'],
                'target': [1, 1, 1, 1, 1, 0, 0, 0],
            }
        )
        hierarchy = {'store': {'North': ('N1', 'N2'), 'South': ('S1',)}}
        enc = encoders.TargetEncoder(
            cols=['store'], hierarchy=hierarchy, min_samples_leaf=20, smoothing=10
        )
        return enc.fit(train[['store']], train['target'])

    def test_inference_hierarchy_unseen_leaf_uses_ancestor(self):
        """An unseen leaf attached to a fitted region gets that region's value."""
        enc = self._fit_store_region_encoder()
        north_code = enc._hier_ordinal_encoder.category_mapping[0]['mapping'].loc['North']
        north_value = enc._hier_level_mappings['store'][0].loc[north_code]
        south_code = enc._hier_ordinal_encoder.category_mapping[0]['mapping'].loc['South']
        south_value = enc._hier_level_mappings['store'][0].loc[south_code]

        inference = pd.DataFrame({'store': ['unknown', 'N1', 'S1']})

        # Dictionary form: the new leaf is explicitly attached to North.
        result = enc.transform(inference, hierarchy={'store': {'North': ('unknown',)}})
        self.assertAlmostEqual(north_value, result['store'].iloc[0])
        # Known leaves keep their own fitted encoding.
        known = enc.transform(pd.DataFrame({'store': ['N1', 'S1']}))
        pd.testing.assert_series_equal(
            result['store'].iloc[1:].reset_index(drop=True),
            known['store'].reset_index(drop=True),
        )

        # The same leaf attached to South must fall back to South's value instead.
        south_result = enc.transform(
            pd.DataFrame({'store': ['unknown']}),
            hierarchy=pd.DataFrame({'HIER_store_1': ['South']}),
        )
        self.assertAlmostEqual(south_value, south_result['store'].iloc[0])
        self.assertNotAlmostEqual(
            north_value, south_result['store'].iloc[0]
        )

    def test_inference_hierarchy_dataframe_form(self):
        """Row-aligned HIER_ dataframes are accepted as inference-time hierarchies."""
        enc = self._fit_store_region_encoder()
        north_code = enc._hier_ordinal_encoder.category_mapping[0]['mapping'].loc['North']
        north_value = enc._hier_level_mappings['store'][0].loc[north_code]

        inference = pd.DataFrame({'store': ['unknown', 'N1']})
        hierarchy = pd.DataFrame({'HIER_store_1': ['North', 'North']})
        result = enc.transform(inference, hierarchy=hierarchy)
        self.assertAlmostEqual(north_value, result['store'].iloc[0])

    def test_inference_without_hierarchy_uses_global_default(self):
        """Without hierarchy information unseen leaves keep the global default."""
        enc = self._fit_store_region_encoder()
        result = enc.transform(pd.DataFrame({'store': ['unknown']}))
        self.assertAlmostEqual(enc._mean, result['store'].iloc[0])

    def test_inference_hierarchy_unseen_parent_falls_back_to_global(self):
        """An unseen hierarchy node cannot be used; the global default remains."""
        enc = self._fit_store_region_encoder()
        result = enc.transform(
            pd.DataFrame({'store': ['unknown']}),
            hierarchy=pd.DataFrame({'HIER_store_1': ['Atlantis']}),
        )
        self.assertAlmostEqual(enc._mean, result['store'].iloc[0])

    def test_inference_hierarchy_is_deterministic_across_batches(self):
        """Repeated batches with the same leaf and hierarchy encode identically."""
        enc = self._fit_store_region_encoder()
        batch = pd.DataFrame({'store': ['unknown']})
        first = enc.transform(batch, hierarchy={'store': {'North': ('unknown',)}})
        second = enc.transform(
            batch, hierarchy=pd.DataFrame({'HIER_store_1': ['North']})
        )
        third = enc.transform(batch, hierarchy={'store': {'North': ('unknown',)}})
        self.assertAlmostEqual(first['store'].iloc[0], second['store'].iloc[0])
        self.assertAlmostEqual(first['store'].iloc[0], third['store'].iloc[0])
        # The call-scoped state must not leak onto the fitted encoder.
        self.assertIsNone(enc._inference_hierarchy)

    def test_inference_hierarchy_uses_deepest_available_ancestor(self):
        """At deeper levels the nearest fitted ancestor provides the fallback."""
        train = pd.DataFrame(
            {
                'store': ['N1', 'N1', 'N2', 'N2', 'S1', 'S1'],
                'target': [1, 1, 1, 1, 0, 0],
            }
        )
        hierarchy = {
            'store': {
                'North': {'N-town': ('N1', 'N2')},
                'South': {'S-town': ('S1',)},
            }
        }
        enc = encoders.TargetEncoder(
            cols=['store'], hierarchy=hierarchy, min_samples_leaf=20, smoothing=10
        ).fit(train[['store']], train['target'])

        north_code = enc._hier_ordinal_encoder.category_mapping[0]['mapping'].loc['North']
        ntown_code = enc._hier_ordinal_encoder.category_mapping[1]['mapping'].loc['N-town']
        north_value = enc._hier_level_mappings['store'][0].loc[north_code]
        ntown_value = enc._hier_level_mappings['store'][1].loc[ntown_code]

        inference = pd.DataFrame({'store': ['new_north', 'new_lost']})
        inference_hierarchy = pd.DataFrame(
            {
                'HIER_store_1': ['North', 'North'],
                'HIER_store_2': ['N-town', 'Atlantis'],
            }
        )
        result = enc.transform(inference, hierarchy=inference_hierarchy)
        # Deepest fitted ancestor (N-town) is used when available...
        self.assertAlmostEqual(ntown_value, result['store'].iloc[0])
        # ...otherwise the search continues up to the next fitted ancestor (North).
        self.assertAlmostEqual(north_value, result['store'].iloc[1])

    def test_inference_hierarchy_resolves_unknown_under_error_strategy(self):
        """With handle_unknown='error', a hierarchy-resolved leaf is accepted."""
        enc = self._fit_store_region_encoder()
        enc.handle_unknown = 'error'
        inference = pd.DataFrame({'store': ['unknown', 'N1']})
        hierarchy = pd.DataFrame({'HIER_store_1': ['North', 'North']})
        result = enc.transform(inference, hierarchy=hierarchy)

        north_code = enc._hier_ordinal_encoder.category_mapping[0]['mapping'].loc['North']
        north_value = enc._hier_level_mappings['store'][0].loc[north_code]
        self.assertAlmostEqual(north_value, result['store'].iloc[0])

        # An unseen parent still cannot resolve the leaf and must raise.
        with self.assertRaises(ValueError):
            enc.transform(
                pd.DataFrame({'store': ['unknown']}),
                hierarchy=pd.DataFrame({'HIER_store_1': ['Atlantis']}),
            )

    def test_inference_hierarchy_without_fitted_hierarchy_raises(self):
        """Passing an inference hierarchy to a hierarchy-free fit is an error."""
        train = pd.DataFrame({'store': ['a', 'a', 'b'], 'target': [1, 0, 1]})
        enc = encoders.TargetEncoder(cols=['store']).fit(
            train[['store']], train['target']
        )
        with self.assertRaises(ValueError):
            enc.transform(
                pd.DataFrame({'store': ['c']}),
                hierarchy={'store': {'a': ('c',)}},
            )

    def test_hierarchy_with_scikit_learn_column_transformer(self) -> None:
        """Test that the encoder works with a scikit-learn ColumnTransformer."""
        features: list[str] = ["cat_feature", "int_feature"]
        target: str = "target"

        df: pd.DataFrame = pd.DataFrame({
            "cat_feature": ["aa", "ab", "ac", "ba", "bb", "bc"],
            "int_feature": [10, 11, 9, 8, 12, 10],
            "target": [1, 2, 1, 2, 1, 2]
        })
        hierarchical_map: dict = {
            "cat_feature": {
                "a": ("aa", "ab", "ac"),
                "b": ("ba", "bb", "bc"),
            },
        }

        # Get the encoder values from a simple TargetEncoder
        target_encoder = encoders.TargetEncoder(cols=["cat_feature"], min_samples_leaf=2, smoothing=2, hierarchy=hierarchical_map)
        encoder_values = target_encoder.fit_transform(df[features], df[target])["cat_feature"].values

        # Get the encoder values from a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", encoders.TargetEncoder(cols=["cat_feature"], min_samples_leaf=2, smoothing=2, hierarchy=hierarchical_map), ["cat_feature"]),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        columntransformer_encoder_values = preprocessor.fit_transform(df[features], df[target])["cat_feature"].values

        # Compare the results
        np.testing.assert_array_almost_equal(encoder_values, columntransformer_encoder_values, decimal=4)
