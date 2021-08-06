import pandas as pd
from typing import List

__all__ = ['norm_df_cols']

def norm_df_cols(df:pd.DataFrame, cols: List[str]):
    r"""Normalize the row across multiple columns. Change the data frame inplace."""
    sum_col = df[cols].sum(axis=1)
    sum_col[sum_col == 0] = 1 # Prevent divided by zero
    for col in cols:
        df[col] = df[col].astype('float') / sum_col
    return df