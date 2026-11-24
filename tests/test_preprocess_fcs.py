import os
import tempfile
import unittest
from unittest.mock import patch

import flowio
import numpy as np
import pandas as pd

import preprocess_fcs


class PreprocessFcsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.input_path = os.path.join(self.temp.name, 'input')
        self.output_path = os.path.join(self.temp.name, 'output')
        os.mkdir(self.input_path)
        self.config = {
            'path_unsup_training': self.input_path,
            'save_path_unsup_training': self.output_path,
            'size_per_sample': 3,
            'export_uncompensated_fcs': True,
            'val_range': [0, 1000],
            'delete_channels': [],
            'harmonize_channels': False,
            'trafo_arcsinh': True,
            'multiply_FSSS': True,
        }

    def write_input_fcs(self):
        values = np.array([[10, 10], [20, 10], [30, 10], [40, 10], [50, 10]])
        with open(os.path.join(self.input_path, 'sample.fcs'), 'wb') as output:
            flowio.create_fcs(
                output, values.ravel().tolist(), ['A', 'B'],
                metadata_dict={'SPILLOVER': '2,A,B,1,0.5,0,1'},
            )
        return values

    def read_output(self, filename):
        data = flowio.FlowData(os.path.join(self.output_path, filename))
        return data, np.array(data.events).reshape(data.event_count, data.channel_count)

    def test_roundtrip_compensated_and_uncompensated_same_events(self):
        original = self.write_input_fcs()
        self.config.update(harmonize_channels=True, all_channels=['X', 'Y'])
        preprocess_fcs.preprocess(self.config)

        comp, comp_events = self.read_output('sample.fcs_compensated.fcs')
        raw, raw_events = self.read_output('sample.fcs_uncompensated.fcs')
        self.assertNotIn('spillover', comp.text)
        self.assertEqual(raw.text['spillover'], '2,X,Y,1.0,0.5,0.0,1.0')
        self.assertEqual(raw.text['p3n'], 'event_nr')
        self.assertEqual(len(raw_events), 3)
        np.testing.assert_array_equal(comp_events[:, 2], raw_events[:, 2])
        indices = raw_events[:, 2].astype(int) - 1
        np.testing.assert_array_equal(raw_events[:, :2], original[indices])
        expected = np.linalg.solve(np.array([[1, 0.5], [0, 1]]).T, original[indices].T).T
        np.testing.assert_allclose(comp_events[:, :2], expected)

    def test_csv_without_matrix_and_option_disabled(self):
        pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).to_csv(
            os.path.join(self.input_path, 'sample.csv'), index=False
        )
        preprocess_fcs.preprocess(self.config)
        _, events = self.read_output('sample.csv_compensated.fcs')
        np.testing.assert_array_equal(events, [[1, 3, 1], [2, 4, 2]])
        self.assertFalse(os.path.exists(os.path.join(self.output_path, 'sample.csv_uncompensated.fcs')))

        self.write_input_fcs()
        self.config['export_uncompensated_fcs'] = False
        preprocess_fcs.preprocess(self.config)
        self.assertFalse(os.path.exists(os.path.join(self.output_path, 'sample.fcs_uncompensated.fcs')))

    def test_rejects_removing_spill_channel_for_uncompensated_export(self):
        self.write_input_fcs()
        self.config['delete_channels'] = ['A']
        with self.assertRaisesRegex(ValueError, 'cannot delete channels'):
            preprocess_fcs.preprocess(self.config)

    def test_compensation_failure_is_not_exported_as_success(self):
        with open(os.path.join(self.input_path, 'singular.fcs'), 'wb') as output:
            flowio.create_fcs(
                output, [1, 2], ['A', 'B'],
                metadata_dict={'SPILLOVER': '2,A,B,1,1,1,1'},
            )
        with self.assertRaisesRegex(ValueError, 'compensation failed'):
            preprocess_fcs.preprocess(self.config)
        self.assertFalse(os.path.exists(os.path.join(self.output_path, 'singular.fcs_compensated.fcs')))

    def test_summary_records_options_and_actual_counts_for_each_file(self):
        self.write_input_fcs()
        pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).to_csv(
            os.path.join(self.input_path, 'sample.csv'), index=False
        )
        self.config.update(harmonize_channels=True, all_channels=['X', 'Y'])
        preprocess_fcs.preprocess(self.config)

        summaries = [name for name in os.listdir(self.output_path) if name.endswith('.txt')]
        self.assertEqual(len(summaries), 1)
        with open(os.path.join(self.output_path, summaries[0]), encoding='utf-8') as file:
            summary = file.read()
        self.assertIn('Downsampling target per file: 3', summary)
        self.assertIn('Uncompensated FCS export requested: True', summary)
        self.assertIn("Harmonized channels: ['X', 'Y']", summary)
        self.assertIn("'sample.fcs': raw events=5, retained events=3, "
                      "compensation applied=True, uncompensated FCS exported=True", summary)
        self.assertIn("'sample.csv': raw events=2, retained events=2, "
                      "compensation applied=False, uncompensated FCS exported=False", summary)

        self.config['export_uncompensated_fcs'] = False
        preprocess_fcs.preprocess(self.config)
        summaries = [name for name in os.listdir(self.output_path) if name.endswith('.txt')]
        self.assertEqual(len(summaries), 2)
        latest = max(summaries)
        with open(os.path.join(self.output_path, latest), encoding='utf-8') as file:
            summary = file.read()
        self.assertIn('Uncompensated FCS export requested: False', summary)
        self.assertIn("'sample.fcs': raw events=5, retained events=3, "
                      "compensation applied=True, uncompensated FCS exported=False", summary)

    def test_main_reads_selected_yaml(self):
        with patch.object(preprocess_fcs, 'preprocess') as process:
            preprocess_fcs.main()
        self.assertIn('path_unsup_training', process.call_args.args[0])
        self.assertIn('export_uncompensated_fcs', process.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
