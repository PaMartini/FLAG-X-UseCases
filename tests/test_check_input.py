import os
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from anndata import AnnData

import check_input
from import_lmd_stubb import add_pns_to_fcs30


class CheckInputTests(unittest.TestCase):
    def test_csv_formats_and_quoted_delimiters(self):
        examples = [
            ('english.csv', '"CD, marker",CD3\n1.5,2.5\n', ['CD, marker', 'CD3']),
            ('german.csv', '"CD; marker";CD3\n1,5;2,5\n', ['CD; marker', 'CD3']),
            ('single.csv', 'channel\n1,5\n', ['channel']),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for name, content, expected in examples:
                path = os.path.join(directory, name)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                with self.subTest(name=name):
                    self.assertEqual(check_input.input_details(directory, name), (expected, None))

    def test_ambiguous_and_empty_csv_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, content in [('ambiguous.csv', 'a,b;c\n'), ('empty.csv', '')]:
                path = os.path.join(directory, name)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    check_input.input_details(directory, name)

    def test_fcs_and_lmd_matrix_presence(self):
        for extension in ('fcs', 'lmd'):
            for spill_key in (None, 'spill', 'spillover'):
                data = SimpleNamespace(version='3.0', text={'p1n': 'FL1-A', 'p1s': 'CD3'})
                if spill_key:
                    data.text[spill_key] = '1,FL1-A,1'
                fcs2 = SimpleNamespace(version='2.0', text={'p1n': 'FL1 INT LOG', 'p1s': 'CD3'})
                datasets = [fcs2, data] if extension == 'lmd' else [data]
                with self.subTest(extension=extension, spill_key=spill_key), \
                        patch('flowio.read_multiple_data_sets', return_value=datasets) as read:
                    self.assertEqual(
                        check_input.input_details('/input', f'sample.{extension}'),
                        (['CD3'], spill_key is not None),
                    )
                    read.assert_called_once_with(f'/input/sample.{extension}', only_text=True)

    def test_fcs_duplicate_pns_falls_back_to_pnn(self):
        data = SimpleNamespace(text={'p1n': 'FL1', 'p1s': 'CD3', 'p2n': 'FL2', 'p2s': 'CD3'})
        with patch('flowio.read_multiple_data_sets', return_value=[data]):
            self.assertEqual(check_input.input_details('/input', 'sample.fcs'), (['FL1', 'FL2'], False))

    def test_lmd_fallback_and_missing_version(self):
        fcs2 = SimpleNamespace(version='2.0', text={'p1n': 'FL1 INT LOG'})
        fcs3 = SimpleNamespace(version='3.0', text={'p1n': 'FL1-A'})
        with patch('flowio.read_multiple_data_sets', return_value=[fcs2, fcs3]):
            self.assertEqual(check_input.input_details('/input', 'sample.lmd'), (['FL1-A'], False))
        with patch('flowio.read_multiple_data_sets', return_value=[fcs3]), self.assertRaisesRegex(ValueError, '2.0'):
            check_input.input_details('/input', 'sample.lmd')

    def test_lmd_mapping_uses_channel_identifiers(self):
        fcs2 = SimpleNamespace(version='2.0', text={
            'p1n': 'FS INT LIN', 'p1s': 'FS INT LIN',
            'p2n': 'FL1 INT LOG', 'p2s': 'CD3',
            'p3n': 'FS PEAK LIN', 'p3s': 'FS PEAK LIN',
            'p4n': 'FL4 INT LOG', 'p4s': '',
        })
        fcs3 = SimpleNamespace(version='3.0', text={
            'p1n': 'FS-H', 'p2n': 'FS-A', 'p3n': 'FL1-A', 'p4n': 'FL4-A', 'p5n': 'OTHER',
        })
        with patch('flowio.read_multiple_data_sets', return_value=[fcs2, fcs3]):
            self.assertEqual(
                check_input.input_details('/input', 'sample.lmd'),
                (['FS PEAK', 'FS INT', 'CD3', 'FL4 INT', 'OTHER'], False),
            )

    def test_malformed_spillover_is_reported(self):
        data = SimpleNamespace(text={'p1n': 'FL1', 'spillover': 'not a matrix'})
        with patch('flowio.read_multiple_data_sets', return_value=[data]), self.assertRaises(ValueError):
            check_input.input_details('/input', 'sample.fcs')

    def test_lmd_loader_without_spillover_matrix(self):
        def make_sample(pnn, pns=None):
            var = pd.DataFrame({'PnN': [pnn]})
            if pns is not None:
                var['PnS'] = [pns]
            return AnnData(
                X=np.zeros((1, 1)),
                var=var,
                uns={'filename': 'sample.LMD', 'meta': {'spill': None}},
            )

        fcs3 = SimpleNamespace(
            anndata_list_=[make_sample('FL1-A')],
            load_data_files_to_anndata=lambda **kwargs: None,
        )
        fcs2 = SimpleNamespace(
            anndata_list_=[make_sample('FL1 INT LOG', 'CD3')],
            load_data_files_to_anndata=lambda **kwargs: None,
        )
        with patch('flagx.io.FlowDataManager', side_effect=[fcs3, fcs2]):
            loaded = add_pns_to_fcs30('/input', ['sample.LMD'], verbosity=0)
        self.assertEqual(loaded.anndata_list_[0].var_names.tolist(), ['CD3'])
        self.assertIsNone(loaded.anndata_list_[0].uns['meta']['spill'])

    def test_summary_from_csv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = os.path.join(directory, 'input')
            output_dir = os.path.join(directory, 'output')
            os.mkdir(input_dir)
            for name, header in [('a.csv', 'A,B'), ('b.csv', 'B,A'), ('c.csv', 'A,C')]:
                with open(os.path.join(input_dir, name), 'w', encoding='utf-8') as f:
                    f.write(header + '\n1,2\n')
            with patch.object(check_input, 'input_path', input_dir), patch.object(check_input, 'output_path', output_dir):
                check_input.main()
            with open(os.path.join(output_dir, os.listdir(output_dir)[0]), encoding='utf-8') as f:
                report = f.read()
            self.assertTrue(report.endswith(
                "\nChannel combinations (this run; channel order ignored):\n"
                "2 files with channel combination: ['A', 'B']\n"
                "1 files with channel combination: ['A', 'C']\n"
            ))

    def test_report_includes_supported_files_and_ignores_other_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = os.path.join(directory, 'input')
            output_dir = os.path.join(directory, 'output')
            os.mkdir(input_dir)
            for name in ['a.CSV', 'b.FCS', 'c.LMD', 'notes.txt']:
                with open(os.path.join(input_dir, name), 'w', encoding='utf-8') as f:
                    f.write('FS INT,CD3\n1.5,2.5\n')
            os.mkdir(os.path.join(input_dir, 'folder.csv'))
            with (
                patch.object(check_input, 'input_path', input_dir),
                patch.object(check_input, 'output_path', output_dir),
                patch.object(
                    check_input, 'input_details',
                    side_effect=[
                        (['FS INT', 'CD3'], None),
                        (['CD3', 'FS INT'], True),
                        (['FS INT', 'CD4'], False),
                    ] * 2,
                ) as load,
                patch.object(check_input, 'datetime') as clock,
            ):
                clock.now.return_value = datetime(2026, 11, 7, 16, 58)
                check_input.main()
                check_input.main()
            self.assertEqual(
                [call.args[1] for call in load.call_args_list],
                ['a.CSV', 'b.FCS', 'c.LMD'] * 2,
            )
            with open(os.path.join(output_dir, os.listdir(output_dir)[0]), encoding='utf-8') as f:
                self.assertEqual(
                    f.read(),
                    "check_input_2026-11-07_16-58 list of files and channels\n"
                    + (
                        "a.CSV\n['FS INT', 'CD3']\n"
                        "b.FCS\nwith compensation matrix\n['CD3', 'FS INT']\n"
                        "c.LMD\nno compensation matrix\n['FS INT', 'CD4']\n"
                        "\nChannel combinations (this run; channel order ignored):\n"
                        "2 files with channel combination: ['FS INT', 'CD3']\n"
                        "1 files with channel combination: ['FS INT', 'CD4']\n"
                    ) * 2,
                )


if __name__ == '__main__':
    unittest.main()
