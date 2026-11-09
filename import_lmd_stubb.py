# helper function to import lmd. Modified 2026-11-04

import re

import pandas as pd


def lmd_channel_mapping(fcs2_var, fcs3_var):
    """Map FCS3.0 PnN names to channel annotations from FCS2.0."""
    def extract_pnn_id(pnn_name):
        name = pnn_name.strip().upper()
        match = re.match(r'^(FL\d+)', name)
        if match:
            return match.group(1)
        if name.startswith('TIME'):
            return 'TIME'
        match = re.match(r'^(FS|SS)(?:[- ]?(INT|PEAK|TOF|A|H|W))?', name)
        if match:
            suffix = {None: 'INT', 'A': 'INT', 'H': 'PEAK', 'W': 'TOF'}.get(match.group(2), match.group(2))
            return f'{match.group(1)} {suffix}'
        return None

    pnn_id_to_pns = {}
    for _, row in fcs2_var.iterrows():
        pnn_id = extract_pnn_id(row['PnN'])
        pns = row['PnS']
        if pnn_id is None:
            continue
        if pd.isna(pns) or str(pns).strip() in ('', str(row['PnN']).strip()):
            pns = re.sub(r'\s+(LIN|LOG)$', '', row['PnN'].strip(), flags=re.IGNORECASE)
        pnn_id_to_pns[pnn_id] = pns

    return {
        pnn: pnn_id_to_pns.get(extract_pnn_id(pnn), pnn)
        for pnn in fcs3_var['PnN']
    }

def add_pns_to_fcs30(data_path, lmd_fns, verbosity=1, save_path=None):
    """
    LMD File PnS Annotation Transfer Script
    ========================================

    This script loads LMD files and transfers PnS annotations from the FCS2.0 compliant part 
    to the FCS3.0 compliant part of the same files.

    Background
    ----------
    LMD files typically contain both FCS2.0 and FCS3.0 compliant data sections. The FCS3.0 section
    uses generic channel names (e.g. 'FL1-A', 'FL2-A') in its PnN field, while the FCS2.0
    section contains the actual marker/antibody names (e.g. '15-FITC', '13-PE') in its PnS field.
    This script bridges the two by extracting the PnS annotations from FCS2.0 and mapping them
    to the corresponding FCS3.0 channels via a shared channel identifier (e.g. 'FL1', 'FS', 'SS', 'TIME').

    Workflow
    --------
    1. Load the FCS3.0 compliant part of LMD files into AnnData objects via FlowDataManager.
    2. Load the FCS2.0 compliant part of the same LMD files into a helper FlowDataManager.
    3. For each FCS3.0 AnnData object:
       a. Find the matching FCS2.0 AnnData object by filename.
       b. Build a mapping from channel identifier (e.g. 'FL1') to PnS (e.g. '15-FITC') using the FCS2.0 PnN and PnS columns.
       c. Map the FCS3.0 PnN names (e.g. 'FL1-A') to PnS via the shared channel identifier.
       d. Set PnS as the index of adata.var, retaining it as a column.
       e. Rename the spillover matrix index and columns to PnS if present.
    4. Delete the helper FlowDataManager to free memory.

    Channel Identifier Extraction
    ------------------------------
    The helper function ``extract_pnn_id`` extracts a normalized channel identifier from
    a raw PnN string using the following rules:

    - FL<n>  : matches 'FL1 INT LOG', 'FL1-A', 'FL10-A' etc.  -> 'FL1', 'FL10'
    - FS     : matches 'FS INT LIN', 'FS-A' etc.              -> 'FS'
    - SS     : matches 'SS INT LIN', 'SS-A' etc.              -> 'SS'
    - TIME   : matches 'TIME'                                  -> 'TIME'

    Example
    -------
    FCS2.0 PnN 'FL1 INT LOG' and FCS3.0 PnN 'FL1-A' both resolve to the identifier 'FL1',
    which is then used to look up the PnS value '15-FITC' from the FCS2.0 section.

    """

    import gc
    import warnings
    from flagx.io import FlowDataManager

    # --- Load FCS3.0 compliant part of LMD files
    fdm = FlowDataManager(
        data_file_names=lmd_fns,
        data_file_path=data_path,
        save_path=save_path,
        verbosity=verbosity,
    )
    # LMD spillover matrices are index-encoded; the loader resolves this itself but warns about it
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Spillover columns do not match PnN')
        fdm.load_data_files_to_anndata(fcs_version_lmd='3.0')

    # --- Load FCS2.0 compliant part of LMD files
    fdm_helper = FlowDataManager(
        data_file_names=lmd_fns,
        data_file_path=data_path,
        verbosity=0,
    )
    fdm_helper.load_data_files_to_anndata(fcs_version_lmd='2.0')


    # --- Extract PnS annotations from FCS2.0 and add to FCS3.0

    # Iterate over FCS3.0 samples/AnnData objects
    for adata in fdm.anndata_list_:
        fn = adata.uns['filename']

        # Iterate over FCS2.0 samples/AnnData objects
        for bdata in fdm_helper.anndata_list_:

            # Skip if filenames do not match
            if bdata.uns['filename'] != fn:
                continue

            # Check if PnS is present in FCS2.0
            if not 'PnS' in bdata.var.columns:
                continue

            pnn_fcs3_to_pns = lmd_channel_mapping(bdata.var, adata.var)

            # Spillover labels are decoded positionally by the loader, so map old var names -> PnS by position
            old_to_pns = dict(zip(adata.var_names, adata.var['PnN'].map(pnn_fcs3_to_pns)))

            # Apply mapping
            adata.var['PnS'] = adata.var['PnN'].map(pnn_fcs3_to_pns)

            # Reindex with PnS
            adata.var.set_index('PnS', inplace=True, drop=False)

            # Reindex spillover matrix if present
            spill_mat = adata.uns['meta'].get('spill')
            if spill_mat is not None:
                spill_mat.index = spill_mat.index.map(lambda x: old_to_pns.get(x, x))
                spill_mat.columns = spill_mat.columns.map(lambda x: old_to_pns.get(x, x))
                adata.uns['meta']['spill'] = spill_mat

            break

    # Delete the helper data from memory
    del fdm_helper
    gc.collect()

    return fdm



if __name__ == '__main__':

    add_pns_to_fcs30(
        data_path='/home/paulm/projects/mrd_detection/data/old_AL_Controls_and_AML',  # Todo: Adjust path
        lmd_fns=sorted(['AL1 KMED 2016_338 maessig_atypisch.LMD', ]),  # Todo: Adjust filenames
    )

    print('done')
























