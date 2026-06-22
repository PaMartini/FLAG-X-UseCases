

def add_pns_to_fcs30():
    """
    LMD File PnS Annotation Transfer Script
    ========================================

    This script loads LMD files and transfers PnS annotations from the FCS2.0 compliant part to the FCS3.0 compliant part of the same files.

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
    import re
    from flagx.io import FlowDataManager

    data_path = '/home/paulm/projects/mrd_detection/data/old_AL_Controls_and_AML'  # Todo: Adjust path
    lmd_fns = sorted(['AL1 KMED 2016_338 maessig_atypisch.LMD', ])  # Todo: Adjust filenames

    # --- Load FCS3.0 compliant part of LMD files
    fdm = FlowDataManager(
        data_file_names=lmd_fns,
        data_file_path=data_path,
    )
    fdm.load_data_files_to_anndata(fcs_version_lmd='3.0')

    # --- Load FCS2.0 compliant part of LMD files
    fdm_helper = FlowDataManager(
        data_file_names=lmd_fns,
        data_file_path=data_path,
    )
    fdm_helper.load_data_files_to_anndata(fcs_version_lmd='2.0')


    # --- Implement helper function to extract identifier of PnN channel name based on its first characters
    def extract_pnn_id(pnn_name: str) -> str | None:
        """Extract FL<n>, FS, SS, or TIME prefix from a channel name."""
        name = pnn_name.strip()
        if re.match(r'^FL\d+', name, re.IGNORECASE):
            m = re.match(r'^(FL\d+)', name, re.IGNORECASE)
            return m.group(1).upper()
        for prefix in ('FS', 'SS', 'TIME'):
            if name.upper().startswith(prefix):
                return prefix
        return None


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

            # Build mapper: FLi/FS/SS/TIME -> PnS  (from FCS2.0 PnN column)
            # FCS2.0 PnN contains e.g. "FL1 INT LOG"; extract "FL1" as the key
            # FCS3.0 PnN contains e.g. "FL1-A"; extract "FL1" as the key
            pnn_id_to_pns = {}
            for _, row in bdata.var.iterrows():
                pns_fcs2 = row['PnS']
                pnn_fcs2 = row['PnN']
                pnn_id = extract_pnn_id(pnn_fcs2)
                pnn_id_to_pns[pnn_id] = pns_fcs2

            # Map FCS3.0 PnN -> PnS using the extracted identifiers
            pnn_fcs3_to_pns = {}
            for _, row in adata.var.iterrows():
                pnn = row['PnN']
                pnn_id = extract_pnn_id(pnn_name=pnn)
                pnn_fcs3_to_pns[pnn] = pnn_id_to_pns.get(pnn_id, pnn)

            # Apply mapping
            adata.var['PnS'] = adata.var['PnN'].map(pnn_fcs3_to_pns)

            # Reindex with PnS
            adata.var.set_index('PnS', inplace=True, drop=False)

            # Reindex spillover matrix if present
            if 'spill' in adata.uns['meta'].keys():
                spill_mat = adata.uns['meta']['spill']
                spill_mat.index = spill_mat.index.map(pnn_fcs3_to_pns)
                spill_mat.columns = spill_mat.columns.map(pnn_fcs3_to_pns)
                adata.uns['meta']['spill'] = spill_mat

            break

    # Delete the helper data from memory
    del fdm_helper
    gc.collect()

    # Todo: Any downstream task using the re-indexed FCS3.0 data ...



if __name__ == '__main__':

    add_pns_to_fcs30()

    print('done')


























