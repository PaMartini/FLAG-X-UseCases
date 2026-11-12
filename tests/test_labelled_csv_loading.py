import os
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from flagx.io import FlowDataManager


class LoadingComplete(Exception):
    pass


class LabelledCsvLoadingTests(unittest.TestCase):
    def test_mixed_formats_are_loaded_numerically_in_order(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, 'config_Imstat.yml'), encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with tempfile.TemporaryDirectory() as directory:
            config['path_sup_training'] = directory
            config['save_path_sup_training'] = os.path.join(directory, 'results')
            for name, contents in [
                ('a.csv', 'channel,population,sample_idx\n1.5,1,10\n'),
                ('b.csv', 'channel;population;sample_idx\n2,75;2;20\n'),
            ]:
                with open(os.path.join(directory, name), 'w', encoding='utf-8') as f:
                    f.write(contents)

            loaded = []
            dataframes = []
            read_csv = pd.read_csv

            def record_read(*args, **kwargs):
                dataframe = read_csv(*args, **kwargs)
                dataframes.append(dataframe)
                return dataframe

            def stop_after_loading(manager, **kwargs):
                self.assertEqual(manager.data_file_type, 'csv')
                loaded.extend(manager.anndata_list_)
                raise LoadingComplete()

            with patch('yaml.safe_load', return_value=config), \
                    patch('pandas.read_csv', side_effect=record_read), \
                    patch.object(FlowDataManager, 'check_sample_sizes', stop_after_loading), \
                    self.assertRaises(LoadingComplete):
                runpy.run_path(os.path.join(root, 'training_labelled_csv_vs2.py'))

            self.assertEqual([a.uns['filename'] for a in loaded], ['a.csv', 'b.csv'])
            self.assertEqual([a.var_names.tolist() for a in loaded], [
                ['channel', 'population', 'sample_idx'],
                ['channel', 'population', 'sample_idx'],
            ])
            np.testing.assert_array_equal(loaded[0].X, [[1.5, 1, 10]])
            np.testing.assert_array_equal(loaded[1].X, [[2.75, 2, 20]])
            np.testing.assert_array_equal(dataframes[0].to_numpy(), [[1.5, 1, 10]])
            np.testing.assert_array_equal(dataframes[1].to_numpy(), [[2.75, 2, 20]])

