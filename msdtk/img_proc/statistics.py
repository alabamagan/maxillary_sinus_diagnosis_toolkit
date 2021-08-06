import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from pytorch_med_imaging.med_img_dataset import ImageDataSet
import tqdm.auto as auto
import pandas as pd
import SimpleITK as sitk

__all__ = ['label_statistics']

def label_statistics(label_dir: str,
                     id_globber: str = "^[0-9]+_(L|R)",
                     num_workers: int = 8,
                     verbose: bool = True,
                     normalized: bool = False) -> pd.DataFrame:
    r"""Return the data statistics of the labels"""
    # Prepare torchio sampler
    labelimages = ImageDataSet(label_dir, verbose=verbose, dtype='uint8', idGlobber=id_globber)

    out_df = pd.DataFrame()
    for i, s in enumerate(auto.tqdm(labelimages.data_source_path)):
        s = sitk.ReadImage(s)
        shape_stat = sitk.LabelShapeStatisticsImageFilter()
        shape_stat.Execute(s)
        val = shape_stat.GetLabels()

        #-------------
        # Volume
        #-------------
        names = [f'Volume_{a}' for a in val]
        data = np.asarray([shape_stat.GetNumberOfPixels(v) for v in val])

        # Calculate null labels
        total_counts = np.prod(s.GetSize())
        null_count = total_counts - data.sum()

        names = np.concatenate([['Volume_0'], names])
        data = np.concatenate([[null_count], data])

        # normalizem exclude null label
        if normalized:
            data = data / data[1:].sum()


        #--------------
        # Roundness
        #--------------
        names = np.concatenate([names, [f'Roundness_{a}' for a in val]])
        roundness = [shape_stat.GetRoundness(int(a)) for a in val]
        data = np.concatenate([data, np.asarray(roundness)])

        #--------------
        # Perimeter
        #--------------
        names = np.concatenate([names, [f'Perimeter_{a}' for a in val]])
        perim = np.asarray([shape_stat.GetPerimeter(a) for a in val])
        data = np.concatenate([data, perim])

        row = pd.Series(data = data.tolist(), index=names, name=labelimages.get_unique_IDs()[i])
        out_df = out_df.join(row, how='outer')

    out_df.fillna(0, inplace=True)
    out_df = out_df.T

    # Compute sum of counts
    dsum = out_df.sum()
    davg = out_df.mean()
    dsum.name = 'sum'
    davg.name = 'avg'

    out_df = out_df.append([dsum, davg])
    out_df.index.name = 'Patient ID'
    labelimages._logger.info(f"\n{out_df.to_string()}")

    return out_df