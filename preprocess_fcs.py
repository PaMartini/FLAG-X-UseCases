# Preprocesses a list of individual flow cytometry files.
# Adds an `event_nr` counter column. This may be useful downstream to list events in populations of interest.
# Optional downsampling and harmonization of channels.
# Export modified files as FCS, always compensated version and optionally uncompensated version including spillover information.
# tested 2026-11-24

import os
from datetime import datetime

import flowio
import numpy as np
import yaml
from flagx.io import FlowDataManager

from channel_harmonize_module import prepare_channels
from import_lmd_stubb import add_pns_to_fcs30


# Select the configuration YAML file for this workflow.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_Imstat.yml')


def load_sample(input_path, filename, output_path):
    if filename.lower().endswith('.lmd'):
        manager = add_pns_to_fcs30(
            data_path=input_path, lmd_fns=[filename], save_path=output_path, verbosity=1
        )
    else:
        manager = FlowDataManager(
            data_file_names=[filename],
            data_file_path=input_path,
            save_path=output_path,
            verbosity=1,
        )
        manager.load_data_files_to_anndata()
    if len(manager.anndata_list_) != 1:
        raise ValueError(f'Could not load exactly one sample from {filename!r}')
    return manager


def write_fcs(adata, path, val_range, layer=None, spill=None):
    names = list(adata.var_names) + ['event_nr']
    if len(set(names)) != len(names):
        raise ValueError('Input channel names must be unique and must not include event_nr')
    data = adata.X if layer is None else adata.layers[layer]
    data = np.column_stack((data, adata.obs['event_nr'].to_numpy()))
    metadata = {f'P{i}R': str(val_range[1]) for i in range(1, len(names) + 1)}
    metadata[f'P{len(names)}R'] = str(max(val_range[1], int(adata.obs['event_nr'].max()) + 1))
    if spill is not None:
        spill_names = list(spill.columns)
        if list(spill.index) != spill_names or not set(spill_names).issubset(names):
            raise ValueError('Spillover matrix channels do not match the output channels')
        if any(',' in name for name in spill_names):
            raise ValueError('Spillover channel names cannot contain commas')
        metadata['SPILLOVER'] = ','.join(
            [str(len(spill_names)), *spill_names, *(str(value) for value in spill.to_numpy().ravel())]
        )
    with open(path, 'wb') as output:
        flowio.create_fcs(output, data.ravel().tolist(), names, names, metadata)


def preprocess(config):
    started = datetime.now()
    input_path = config['path_unsup_training']
    output_path = config['save_path_unsup_training']
    target = config['size_per_sample']
    export_uncompensated = config['export_uncompensated_fcs']
    if not isinstance(export_uncompensated, bool):
        raise ValueError('export_uncompensated_fcs must be true or false')
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError('size_per_sample must be a positive integer')
    val_range = tuple(config['val_range'])
    if len(val_range) != 2 or val_range[1] <= 0:
        raise ValueError('val_range must contain a positive upper bound')

    filenames = sorted(
        name for name in os.listdir(input_path)
        if name.lower().endswith(('.fcs', '.lmd', '.csv'))
        and os.path.isfile(os.path.join(input_path, name))
    )
    if not filenames:
        raise FileNotFoundError(f'No FCS, LMD or CSV files found in {input_path}')
    os.makedirs(output_path, exist_ok=True)

    processed = []
    for filename in filenames:
        manager = load_sample(input_path, filename, output_path)
        adata = manager.anndata_list_[0]
        if not adata.var_names.is_unique or 'event_nr' in adata.var_names:
            raise ValueError(f'{filename}: duplicate channel names or existing event_nr channel')
        if adata.n_obs == 0:
            raise ValueError(f'{filename}: input contains no events')
        # FCS float32 cannot represent every integer above 2**24.
        if adata.n_obs > 2**24:
            raise ValueError(f'{filename}: too many events for exact float32 event numbering')
        raw_events = adata.n_obs
        adata.obs['event_nr'] = np.arange(1, adata.n_obs + 1)
        spill = adata.uns.get('meta', {}).get('spill')
        if spill is not None:
            manager.sample_wise_compensation()
            status = manager.compensation_log_.iloc[0]['logs']
            if status != 'compensation applied successfully':
                raise ValueError(f'{filename}: spillover compensation failed: {status}')
        else:
            print(f'{filename}: no spillover matrix; exporting input values without compensation')

        manager.sample_wise_downsampling(data_set='all', target_num_events=target)
        adata = manager.anndata_list_[0]
        original_names = list(adata.var_names)
        adata = prepare_channels(
            [adata], config.get('delete_channels'), config.get('harmonize_channels'),
            config.get('all_channels'),
        )[0]

        output_spill = None
        if export_uncompensated and spill is not None:
            retained_names = [name for name in original_names if name not in (config.get('delete_channels') or [])]
            mapping = dict(zip(retained_names, adata.var_names))
            if not set(spill.index).issubset(mapping) or not set(spill.columns).issubset(mapping):
                raise ValueError(f'{filename}: cannot delete channels used by the spillover matrix')
            output_spill = spill.rename(index=mapping, columns=mapping)

        prefix = os.path.join(output_path, filename)
        write_fcs(adata, prefix + '_compensated.fcs', val_range)
        if output_spill is not None:
            write_fcs(
                adata, prefix + '_uncompensated.fcs', val_range,
                layer='uncompensated', spill=output_spill,
            )
        elif export_uncompensated:
            print(f'{filename}: uncompensated export skipped; no spillover matrix')

        processed.append((filename, raw_events, adata.n_obs, spill is not None, output_spill is not None))

    summary_path = os.path.join(output_path, f'preprocessing_{started:%Y-%m-%d_%Hh%M%S_%f}.txt')
    with open(summary_path, 'x', encoding='utf-8') as summary:
        summary.write(f'Preprocessing summary: {started:%Y-%m-%d %H:%M:%S}\n')
        summary.write(f'Input path: {input_path}\nOutput path: {output_path}\n')
        summary.write(f'Downsampling target per file: {target}\n')
        summary.write('Compensated FCS export: True (spillover applied if available)\n')
        summary.write(f'Uncompensated FCS export requested: {export_uncompensated}\n')
        summary.write(f'Delete channels (if present): {config.get("delete_channels") or []}\n')
        summary.write(f'Harmonize channels: {bool(config.get("harmonize_channels"))}\n')
        if config.get('harmonize_channels'):
            summary.write(f'Harmonized channels: {config.get("all_channels")}\n')
        summary.write(f'FCS value range: {val_range}\n')
        summary.write('\nProcessed files:\n')
        for filename, raw_events, retained_events, compensated, raw_exported in processed:
            summary.write(
                f'{filename!r}: raw events={raw_events}, retained events={retained_events}, '
                f'compensation applied={compensated}, uncompensated FCS exported={raw_exported}\n'
            )
        summary.write(f'\nTotal processing time: {datetime.now() - started}\n')
    print(f'Preprocessing summary saved to {summary_path}')


def main():
    with open(CONFIG_PATH, encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f'{CONFIG_PATH}: expected a YAML mapping')
    preprocess(config)


if __name__ == '__main__':
    main()
