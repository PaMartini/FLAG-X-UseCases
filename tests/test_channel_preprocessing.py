import os
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from anndata import AnnData
from flagx.io import FlowDataManager

from channel_harmonize_module import prepare_channels


def sample(filename, columns, values):
    adata = AnnData(
        X=np.array(values, dtype=float),
        var=pd.DataFrame(index=columns),
        uns={'filename': filename},
    )
    adata.layers['original'] = adata.X.copy()
    return adata


class PrepareChannelsTests(unittest.TestCase):
    def test_deletes_original_names_and_harmonizes_each_sample_by_position(self):
        samples = [
            sample('one.fcs', ['FS PEAK', 'old A', 'old B'], [[99, 1, 2]]),
            sample('two.lmd', ['other A', 'other B'], [[3, 4]]),
        ]
        result = prepare_channels(samples, ['FS PEAK', 'absent'], True, ['A', 'B'])

        for adata, expected in zip(result, ([[1, 2]], [[3, 4]])):
            self.assertEqual(list(adata.var_names), ['A', 'B'])
            np.testing.assert_array_equal(adata.X, expected)
            np.testing.assert_array_equal(adata.layers['original'], expected)
        self.assertEqual(list(samples[0].var_names), ['FS PEAK', 'old A', 'old B'])

    def test_disabled_harmonization_preserves_names(self):
        data = sample('one.csv', ['A', 'remove', 'B'], [[1, 9, 2]])
        result = prepare_channels([data], ['remove'], False, None)
        self.assertEqual(list(result[0].var_names), ['A', 'B'])
        np.testing.assert_array_equal(result[0].X, [[1, 2]])

    def test_invalid_harmonization_config_and_count_are_reported(self):
        data = sample('bad.fcs', ['A', 'B'], [[1, 2]])
        for names in (None, [], ['A', 'A'], ['A', '']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                prepare_channels([data], [], True, names)
        with self.assertRaisesRegex(ValueError, "bad.fcs.*2 channels.*3 names"):
            prepare_channels([data], [], True, ['A', 'B', 'C'])

    def test_training_import_applies_changes_before_preprocessing_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = os.path.join(directory, 'input')
            os.mkdir(data_dir)
            pd.DataFrame({'discard': [20, 30], 'other_discard': [11, 12],
                          'old_a': [1, 2], 'old_b': [3, 4]}).to_csv(
                os.path.join(data_dir, 'one.csv'), index=False
            )
            pd.DataFrame({'discard': [10, 11], 'different_a': [5, 6], 'different_b': [7, 8]}).to_csv(
                os.path.join(data_dir, 'two.csv'), index=False
            )
            config = {
                'save_path_unsup_training': directory,
                'path_unsup_training': data_dir,
                'delete_channels': ['discard', 'other_discard', 'absent', 'discard'],
                'harmonize_channels': True,
                'all_channels': ['A', 'B'],
                'trainchannels': ['A', 'B'],
                'size_per_sample': 10,
                'SOM_dim': [2, 2],
                'SOM_epochs': 1,
                'val_range': [0, 1000],
                'trafo_arcsinh': True,
                'arcsinh_div': 1,
                'multiply_FSSS': False,
                'upscale_val': {},
                'calcTSNE': False,
                'calcPARC': False,
            }
            compensate = FlowDataManager.sample_wise_compensation

            def compensate_with_spillover(manager):
                matrix = pd.DataFrame(
                    [[1.0, 0.5], [0.0, 1.0]],
                    index=['discard', 'old_a'],
                    columns=['discard', 'old_a'],
                )
                manager.anndata_list_[0].uns['meta'] = {'spill': matrix}
                return compensate(manager)

            with patch('yaml.safe_load', return_value=config), \
                    patch.object(FlowDataManager, 'sample_wise_compensation', autospec=True,
                                 side_effect=compensate_with_spillover), \
                    patch('flagx.gating.SOMClassifier') as classifier, \
                    patch('flagx.io.export_to_fcs') as export:
                classifier.return_value.transform.return_value = (None, np.zeros((4, 2)), None, None)
                runpy.run_path(
                    os.path.join(os.path.dirname(__file__), '..', 'training_from_fcs.py'),
                    run_name='__main__',
                )

            exported = export.call_args.kwargs['data_list']
            self.assertEqual([list(adata.var_names) for adata in exported], [['A', 'B'], ['A', 'B']])
            np.testing.assert_array_equal(exported[0].layers['no_trafo'], [[-9, 3], [-13, 4]])
            np.testing.assert_array_equal(exported[1].layers['no_trafo'], [[5, 7], [6, 8]])
            self.assertEqual(classifier.return_value.fit.call_args.kwargs['X'].shape, (4, 2))
            summary_files = [name for name in os.listdir(directory) if name.startswith('fcs_unsup_training_')]
            self.assertEqual(len(summary_files), 1)
            with open(os.path.join(directory, summary_files[0]), encoding='utf-8') as f:
                summary = f.read()
            one = summary.split('"file": "one.csv", \n')[1].split('"sample": 2')[0]
            two = summary.split('"file": "two.csv", \n')[1].split('"large samples')[0]
            self.assertEqual(summary.count('"deleted_columns":'), 1)
            self.assertIn('"deleted_columns": [\'discard\', \'other_discard\']\n', summary)
            self.assertIn('"original channels": [\'discard\', \'other_discard\', \'old_a\', \'old_b\']\n', one)
            self.assertIn('"harmonized channels": [\'A\', \'B\']\n', one)
            self.assertIn('"original channels": [\'discard\', \'different_a\', \'different_b\']\n', two)
            self.assertIn('"harmonized channels": [\'A\', \'B\']\n', two)

    def test_summary_lists_deleted_columns_without_harmonization(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = os.path.join(directory, 'input')
            os.mkdir(data_dir)
            pd.DataFrame({'discard': [9, 10], 'A': [1, 2]}).to_csv(
                os.path.join(data_dir, 'one.csv'), index=False
            )
            config = {
                'save_path_unsup_training': directory,
                'path_unsup_training': data_dir,
                'delete_channels': ['discard', 'absent'],
                'harmonize_channels': False,
                'trainchannels': ['A'],
                'size_per_sample': 10,
                'SOM_dim': [2, 2],
                'SOM_epochs': 1,
                'val_range': [0, 1000],
                'trafo_arcsinh': True,
                'arcsinh_div': 1,
                'multiply_FSSS': False,
                'upscale_val': {},
                'calcTSNE': False,
                'calcPARC': False,
            }
            with patch('yaml.safe_load', return_value=config), \
                    patch('flagx.gating.SOMClassifier') as classifier, \
                    patch('flagx.io.export_to_fcs'):
                classifier.return_value.transform.return_value = (None, np.zeros((2, 2)), None, None)
                runpy.run_path(
                    os.path.join(os.path.dirname(__file__), '..', 'training_from_fcs.py'),
                    run_name='__main__',
                )
            summary_files = [name for name in os.listdir(directory) if name.startswith('fcs_unsup_training_')]
            self.assertEqual(len(summary_files), 1)
            with open(os.path.join(directory, summary_files[0]), encoding='utf-8') as f:
                summary = f.read()
            self.assertIn('"deleted_columns": [\'discard\']\n', summary)
            self.assertNotIn('"original channels":', summary)
            self.assertNotIn('"harmonized channels":', summary)

            config['delete_channels'] = ['absent']
            with patch('yaml.safe_load', return_value=config), \
                    patch('flagx.gating.SOMClassifier') as classifier, \
                    patch('flagx.io.export_to_fcs'):
                classifier.return_value.transform.return_value = (None, np.zeros((2, 2)), None, None)
                runpy.run_path(
                    os.path.join(os.path.dirname(__file__), '..', 'training_from_fcs.py'),
                    run_name='__main__',
                )
            summary_files = [name for name in os.listdir(directory) if name.startswith('fcs_unsup_training_')]
            with open(os.path.join(directory, sorted(summary_files)[-1]), encoding='utf-8') as f:
                latest_summary = f.read().rsplit('"training_files and cell numbers"', 1)[1]
            self.assertEqual(latest_summary.count('"deleted_columns":'), 1)
            self.assertIn('"deleted_columns": none\n', latest_summary)


if __name__ == '__main__':
    unittest.main()
