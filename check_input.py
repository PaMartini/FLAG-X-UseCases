# Check channel names and FCS/LMD compensation matrices without loading event data.
# Useful to check if a list of files contains identical channels and channel names.
# CSV files may use English (comma-separated) or German (semicolon-separated) format.
# Change input_path to select the folder to check.

import csv
import os
from collections import Counter
from datetime import datetime

import pandas as pd

# !! define input and output paths for checking files !!
input_path = './data/testing'
output_path = './results'

def csv_format(path):
    # Inspect the header only, so decimal commas in data rows cannot affect detection.
    with open(path, encoding='utf-8-sig', newline='') as f:
        header = f.readline()
    if not header:
        raise ValueError(f'CSV file is empty: {path}')

    comma_columns = next(csv.reader([header], delimiter=','))
    semicolon_columns = next(csv.reader([header], delimiter=';'))
    if len(comma_columns) > 1 and len(semicolon_columns) > 1:
        raise ValueError(f'Cannot determine CSV delimiter from header in {path}')
    separator = ';' if len(semicolon_columns) > 1 else ','
    decimal = ',' if separator == ';' else '.'
    return {'sep': separator, 'decimal': decimal}


def csv_channels(path):
    return pd.read_csv(path, **csv_format(path), nrows=0).columns.tolist()

def input_details(data_path, filename):
    extension = os.path.splitext(filename)[1].lower()
    if extension == '.csv':
        return csv_channels(os.path.join(data_path, filename)), None
    if extension not in ('.fcs', '.lmd'):
        raise ValueError(f'Unsupported input file: {filename}')

    import flowio
    from flagx.io.io_utils import _flowdata_meta_data_dict_to_df, _spillover_mat_from_str

    datasets = flowio.read_multiple_data_sets(os.path.join(data_path, filename), only_text=True)
    if extension == '.lmd':
        versions = {data.version: data for data in datasets}
        missing = [version for version in ('2.0', '3.0') if version not in versions]
        if missing:
            raise ValueError(f"File '{filename}' is missing FCS version(s): {', '.join(missing)}")
        data = versions['3.0']
        fcs2_var = _flowdata_meta_data_dict_to_df(versions['2.0'].text)
    else:
        data = datasets[0]

    var = _flowdata_meta_data_dict_to_df(data.text)
    pnn = var['PnN']
    if extension == '.lmd':
        if 'PnS' in fcs2_var.columns:
            from import_lmd_stubb import lmd_channel_mapping

            channels = pnn.map(lmd_channel_mapping(fcs2_var, var)).tolist()
        else:
            channels = pnn.tolist()
    elif 'PnS' in var.columns:
        pns = var['PnS'].where(var['PnS'] != '', pnn)
        channels = (pns if pns.is_unique else pnn).tolist()
    else:
        channels = pnn.tolist()

    has_compensation = False
    for key in ('spill', 'spillover'):
        if key in data.text:
            _spillover_mat_from_str(data.text[key])
            has_compensation = True
    return channels, has_compensation


def main():
    input_list = sorted(
        name for name in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, name))
        and name.lower().endswith(('.csv', '.fcs', '.lmd'))
    )
    if not input_list:
        raise FileNotFoundError(f'No CSV, FCS or LMD files found in {input_path}')
    print(input_list)
    os.makedirs(output_path, exist_ok=True)
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = os.path.join(
        output_path, f'check_input_{date_time_str}.txt'
    )
    combination_counts = Counter()
    combination_channels = {}
    with open(report_path, 'a', encoding='utf-8') as f:
        if f.tell() == 0:
            f.write(f"check_input_{date_time_str} list of files and channels\n\n")
        for filename in input_list:
            channels, has_compensation = input_details(input_path, filename)
            combination = frozenset(Counter(channels).items())
            combination_counts[combination] += 1
            combination_channels.setdefault(combination, channels)
            f.write(filename + "\n")
            if has_compensation is not None:
                status = 'with compensation matrix' if has_compensation else 'no compensation matrix'
                f.write(status + "\n")
            f.write(str(channels) + "\n")
        f.write("\nChannel combinations (this run; channel order ignored):\n")
        for combination, count in combination_counts.items():
            f.write(f"{count} files with channel combination: \n"
                    f"{combination_channels[combination]}\n")
    print(f'Channel list saved to {report_path}')

if __name__ == '__main__':
    main()
