import pandas as pd
from pathlib import Path
from typing import List, Union, Iterable, Optional, Pattern

__all__ = ['norm_df_cols', 'get_ids_from_files', 'get_files_with_id']

def norm_df_cols(df:pd.DataFrame, cols: List[str]):
    r"""Normalize the row across multiple columns. Change the data frame inplace."""
    sum_col = df[[cols]].sum(axis=1)
    for col in cols:
        df[col] = df[col].astype('float') / sum_col
    return df

def get_ids_from_files(files: Iterable[Union[str, Path]],
                       idglobber: Optional[Pattern] = "^[0-9]+",
                       return_dict: Optional[bool] = False):
    r"""Get a sort list of substring globbed from the specified files.

    Args:
        files (list of str or str):
            A list of strings that needs globbing. If a single string is given, its treated as a directory and all the
            files in that directory (non-recursively) that has a suffix '.nii.gz' will be processed.
        idglobber (pattern, Optional):
            ID globbing regex.
        return_dict (bool):
            If true, return dictionary where the keys are the ID globbed and the values are the filename where it's
            globbed.

    Returns:
        dict or list
    """
    import os
    import re, fnmatch

    if isinstance(files, str):
        if not Path(files).is_dir():
            raise OSError(f"Specified file is not a directory: {files}")
        files = fnmatch.filter(os.listdir(files), '*.nii.gz')

    out_dict = {}
    for f in files:
        mo = re.search(idglobber, f)
        id = mo.group() if mo is not None else 'Error'

        if id in out_dict:
            print("Warning, there are multiple files with the same ID")
            if not isinstance(out_dict[id], list):
                out_dict[id] = [out_dict[id]]
            out_dict[id].append(f)
        else:
            out_dict[id] = f

    if return_dict:
        return out_dict
    else:
        return list(out_dict.keys())

def get_files_with_id(directory: Union[str, Path],
                      idlist: List[str],
                      idglobber: Optional[Pattern] = "^[0-9]+",
                      return_dict: Optional[bool] = False):
    r"""Get a list of files using ID

    Args:
        directory (str):
            Where you want to get the images from.
        idlist (list or str):
            The desired image IDs.
        idglobber (Optional pattern):
            ID globbing regex.
        recursive (Optional, bool):
            NOT IMPLEMENTED. Whether or not to search recursively.

    Returns:
        list of str
    """
    import os
    import re, fnmatch

    if not Path(directory).is_dir():
        raise OSError(f"Cannot open specified directory: {directory}")

    files = fnmatch.filter(os.listdir(directory), '*.nii.gz')
    files_id = get_ids_from_files(files, idglobber=idglobber, return_dict=True)

    out_dict = {}
    for keys in idlist:
        _val = files_id.get(keys, None)
        if isinstance(_val, (list, tuple)):
            print("Warning, there are multiple files with the same ID.")
        if _val is None:
            print(f"Warning, can't find file with ID {keys} in target folder.")
        out_dict[keys] = _val

    if return_dict:
        return out_dict
    else:
        return list([out_dict[keys] for keys in out_dict])

