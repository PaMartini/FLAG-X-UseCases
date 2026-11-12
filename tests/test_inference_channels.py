import os
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from anndata import AnnData
from flagx.io import FlowDataManager


SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'fcs_inference_w_calc.py')


class InferenceChannelTests(unittest.TestCase):
    def run_inference(self, directory, config, lmd_loader=None):
        for name in ('som_classifier.pkl', 'mlp_classifier.pkl'):
            open(os.path.join(directory, name), 'w').close()

        with patch('yaml.safe_load', return_value=config), \
                patch('flagx.gating.SOMClassifier') as som, \
                patch('flagx.gating.MLPClassifier') as mlp, \
                patch('flagx.io.export_to_fcs') as export, \
                patch('import_lmd_stubb.add_pns_to_fcs30', side_effect=lmd_loader):
            som.load.return_value.predict.side_effect = lambda x: np.ones(len(x), dtype=int)
            mlp.load.return_value.predict.side_effect = lambda x: np.ones(len(x), dtype=int)
            runpy.run_path(SCRIPT, run_name='__main__')
            return export.call_args_list, mlp.load.return_value.predict.call_args_list

    def config(self, directory):
        return {
            'path_inference': os.path.join(directory, 'input'),
            'path_models': directory,
            'save_path_inference': directory,
            'delete_channels': ['drop', 'unused', 'drop'],
            'harmonize_channels': True,
            'all_channels': ['A', 'B'],
            'trainchannels': ['A', 'B'],
            'size_per_sample': 10,
            'max_rows_per_batch': 3,
            'SOM_dim': [2, 2],
            'SOM_epochs': 1,
            'val_range': [0, 1000],
            'trafo_arcsinh': True,
            'arcsinh_div': 1,
            'multiply_FSSS': False,
            'upscale_val': {},
            'calcTSNE': False,
            'show_SOM': False,
            'calcchannels': ['B'],
            'calc_populations': None,
        }

    def summary(self, directory):
        names = [name for name in os.listdir(directory) if name.startswith('fcs_inference_')]
        self.assertEqual(len(names), 1)
        with open(os.path.join(directory, names[0]), encoding='utf-8') as f:
            return f.read()

    def test_multiple_batches_harmonize_before_prediction_statistics_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            os.mkdir(config['path_inference'])
            for name, columns in (
                ('one.csv', {'drop': [9, 9], 'old_a': [1, 2], 'old_b': [3, 4]}),
                ('two.csv', {'different_a': [5, 6], 'different_b': [7, 8]}),
                ('three.csv', {'drop': [9, 9], 'third_a': [11, 12], 'third_b': [13, 14]}),
            ):
                pd.DataFrame(columns).to_csv(os.path.join(config['path_inference'], name), index=False)
            exports, predictions = self.run_inference(directory, config)

            self.assertEqual(len(exports), 2)
            self.assertEqual(len(predictions), 2)
            samples = [adata for call in exports for adata in call.kwargs['data_list']]
            self.assertEqual([adata.uns['filename'] for adata in samples], ['one.csv', 'three.csv', 'two.csv'])
            for adata in samples:
                self.assertEqual(list(adata.var_names), ['A', 'B'])
            np.testing.assert_array_equal(samples[0].layers['no_trafo'], [[1, 3], [2, 4]])
            np.testing.assert_array_equal(samples[1].layers['no_trafo'], [[11, 13], [12, 14]])
            np.testing.assert_allclose(predictions[0].args[0], np.arcsinh([[1, 3], [2, 4], [11, 13], [12, 14]]))
            summary = self.summary(directory)
            self.assertEqual(summary.count('"deleted_channels":'), 1)
            self.assertIn('"deleted_channels": [\'drop\']\n', summary)
            self.assertIn('two.csv\n"original channels": [\'different_a\', \'different_b\']', summary)
            self.assertIn('"harmonized channels": [\'A\', \'B\']', summary)
            result_file = next(name for name in os.listdir(directory) if name.startswith('df_calc_results_'))
            results = pd.read_csv(os.path.join(directory, result_file), sep=';', decimal=',')
            self.assertEqual(dict(zip(results['sample'], results['mlp_1_B_median'])),
                             {'one.csv': 3.5, 'three.csv': 13.5, 'two.csv': 7.5})

    def test_disabled_harmonization_retains_names_and_reports_no_deletions(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            config.update(delete_channels=['absent'], harmonize_channels=False, all_channels=None)
            os.mkdir(config['path_inference'])
            pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).to_csv(
                os.path.join(config['path_inference'], 'one.csv'), index=False
            )
            exports, _ = self.run_inference(directory, config)
            self.assertEqual(list(exports[0].kwargs['data_list'][0].var_names), ['A', 'B'])
            summary = self.summary(directory)
            self.assertIn('"deleted_channels": none\n', summary)
            self.assertNotIn('"original channels":', summary)
            self.assertNotIn('"harmonized channels":', summary)

    def test_lmd_loader_data_are_prepared(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            os.mkdir(config['path_inference'])
            open(os.path.join(config['path_inference'], 'sample.lmd'), 'w').close()
            fdm = FlowDataManager(data_file_names=['sample.lmd'], data_file_path=config['path_inference'])
            spill = pd.DataFrame(
                [[1.0, 0.5], [0.0, 1.0]],
                index=['drop', 'old_a'], columns=['drop', 'old_a'],
            )
            fdm.anndata_list_ = [AnnData(
                X=np.array([[9., 1., 3.], [9., 2., 4.]]),
                var=pd.DataFrame(index=['drop', 'old_a', 'old_b']),
                uns={'filename': 'sample.lmd', 'meta': {'spill': spill}},
            )]
            exports, _ = self.run_inference(directory, config, lmd_loader=lambda **kwargs: fdm)
            result = exports[0].kwargs['data_list'][0]
            self.assertEqual(list(result.var_names), ['A', 'B'])
            np.testing.assert_array_equal(result.layers['no_trafo'], [[-3.5, 3], [-2.5, 4]])
            self.assertIn('"deleted_channels": [\'drop\']', self.summary(directory))

    def test_channel_count_mismatch_fails_with_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            os.mkdir(config['path_inference'])
            pd.DataFrame({'only': [1, 2]}).to_csv(
                os.path.join(config['path_inference'], 'bad.csv'), index=False
            )
            with self.assertRaisesRegex(ValueError, "bad.csv.*1 channels.*2 names"):
                self.run_inference(directory, config)

    def test_downsampling_applies_per_file_to_predictions_statistics_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            config.update(size_per_sample=3, max_rows_per_batch=4)
            os.mkdir(config['path_inference'])
            for name, count in (('one.csv', 7), ('two.csv', 3), ('three.csv', 2)):
                pd.DataFrame({'A': np.arange(count), 'B': np.arange(count) + 10}).to_csv(
                    os.path.join(config['path_inference'], name), index=False
                )
            exports, predictions = self.run_inference(directory, config)

            self.assertEqual(len(exports), 2)
            samples = [adata for call in exports for adata in call.kwargs['data_list']]
            self.assertEqual({adata.uns['filename']: adata.n_obs for adata in samples},
                             {'one.csv': 3, 'two.csv': 3, 'three.csv': 2})
            by_filename = {adata.uns['filename']: adata for adata in samples}
            np.testing.assert_array_equal(by_filename['two.csv'].layers['no_trafo'], [[0, 10], [1, 11], [2, 12]])
            np.testing.assert_array_equal(by_filename['three.csv'].layers['no_trafo'], [[0, 10], [1, 11]])
            for call, prediction in zip(exports, predictions):
                data = call.kwargs['data_list']
                raw = np.concatenate([adata.layers['no_trafo'] for adata in data])
                np.testing.assert_allclose(prediction.args[0], np.arcsinh(raw))
            result_file = next(name for name in os.listdir(directory) if name.startswith('df_calc_results_'))
            results = pd.read_csv(os.path.join(directory, result_file), sep=';', decimal=',')
            self.assertEqual(dict(zip(results['sample'], results['mlp_1'])),
                             {'one.csv': 3, 'two.csv': 3, 'three.csv': 2})
            for _, row in results.iterrows():
                retained = by_filename[row['sample']].layers['no_trafo']
                self.assertEqual(row['mlp_1_B_median'], np.median(retained[:, 1]))
            self.assertIn('"large samples downsampled to": 3\n', self.summary(directory))

    def test_batching_and_compensation_use_downsampled_events(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            config.update(size_per_sample=2, max_rows_per_batch=5)
            os.mkdir(config['path_inference'])
            for name in ('one.csv', 'two.csv', 'three.csv', 'four.csv'):
                pd.DataFrame({'A': np.arange(9), 'B': np.arange(9) + 10}).to_csv(
                    os.path.join(config['path_inference'], name), index=False
                )
            compensate = FlowDataManager.sample_wise_compensation
            observed_counts = []

            def record_compensation(manager):
                observed_counts.append([adata.n_obs for adata in manager.anndata_list_])
                return compensate(manager)

            with patch.object(FlowDataManager, 'sample_wise_compensation', autospec=True,
                              side_effect=record_compensation):
                exports, predictions = self.run_inference(directory, config)
            self.assertEqual(observed_counts, [[2, 2, 2], [2]])
            self.assertEqual([len(call.kwargs['data_list']) for call in exports], [3, 1])
            self.assertEqual([len(call.args[0]) for call in predictions], [6, 2])
            self.assertIn('"processed in" 2 "batches"', self.summary(directory))


if __name__ == '__main__':
    unittest.main()
