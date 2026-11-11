# This module provides a function to harmononize files by removing specified channels in AnnData objects 
# and optionally harmonizing the remaining channels to a predefined list.

def prepare_channels(adata_list, delete_channels, harmonize_channels, all_channels):
    """Remove named channels and optionally rename the remaining channels by position."""
    delete_channels = set(delete_channels or [])
    if harmonize_channels:
        if not isinstance(all_channels, list) or not all_channels:
            raise ValueError('all_channels must be a non-empty list when harmonize_channels is enabled')
        if any(not isinstance(channel, str) or not channel for channel in all_channels):
            raise ValueError('all_channels must contain non-empty channel names')
        if len(set(all_channels)) != len(all_channels):
            raise ValueError('all_channels must contain unique channel names')

    prepared = []
    for adata in adata_list:
        filename = adata.uns.get('filename', '<unknown>')
        keep = [name not in delete_channels for name in adata.var_names]
        remaining_count = sum(keep)
        if harmonize_channels and remaining_count != len(all_channels):
            raise ValueError(
                f"File '{filename}' has {remaining_count} channels after deletion, "
                f"but all_channels contains {len(all_channels)} names."
            )
        result = adata[:, keep].copy()
        if harmonize_channels:
            result.var_names = all_channels
        prepared.append(result)
    return prepared
