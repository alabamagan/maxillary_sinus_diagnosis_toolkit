import os
import re, fnmatch
from tqdm.auto import tqdm

import numpy as np
import SimpleITK as sitk
import multiprocessing as mpi

from pytorch_med_imaging.logger import Logger
from functools import partial
from pathlib import Path

from typing import Union, Iterable, List, Tuple, Optional, Pattern

from ..utils import get_ids_from_files, get_files_with_id

__all__ = ['label_postproc', 'batch_label_postproc']

class BinaryOpenClose(object):
    r"""
    Description:
        SimpleITK filters that perfoms the following operations:

            1) Binary Opening
            2) Binary Closing
            3) Fill hole
    """
    def __init__(self,
                 opening_radius: int,
                 closing_radius: Optional[int] = None,
                 morphological: Optional[bool] = False,
                 ):
        super(BinaryOpenClose, self).__init__()
        if morphological:
            self._opening_filter, self._closing_filter = sitk.BinaryMorphologicalOpeningImageFilter(), \
                                                         sitk.BinaryMorphologicalClosingImageFilter()
        else:
            self._opening_filter, self._closing_filter = sitk.BinaryOpeningByReconstructionImageFilter(), \
                                                         sitk.BinaryClosingByReconstructionImageFilter()

        if closing_radius is None:
            closing_radius = opening_radius

        self._opening_filter.SetKernelRadius(opening_radius)
        self._opening_filter.SetKernelType(sitk.sitkBall if opening_radius> 1 else sitk.sitkBox)
        self._closing_filter.SetKernelRadius(closing_radius)
        self._closing_filter.SetKernelType(sitk.sitkBall if closing_radius > 1 else sitk.sitkBox)

        self._fill_hole = sitk.VotingBinaryHoleFillingImageFilter()
        self._fill_hole.SetForegroundValue(1)
        self._fill_hole.SetMajorityThreshold(1)
        self._fill_hole.SetRadius(2)

    def __call__(self,
                 in_img: sitk.Image):
        return self._fill_hole.Execute(self._closing_filter.Execute(self._opening_filter.Execute(in_img)))

class BinaryVolumeThreshold(object):
    r"""
    Description:
        Remove island with volume smaller than the desired threshold.
    """
    def __init__(self,
                 volume_threshold: float):
        super(BinaryVolumeThreshold, self).__init__()
        self.volume_threshold = volume_threshold

    def __call__(self,
                 in_img: sitk.Image):
        binary_out = sitk.BinaryThreshold(in_img, 1, 255)
        cc_filter = sitk.ConnectedComponentImageFilter()
        cseg = cc_filter.Execute(binary_out)
        n_objs = cc_filter.GetObjectCount()

        if n_objs < 2:
            return in_img

        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(cseg)

        sizes = np.asarray([shape_stats.GetPhysicalSize(i + 1) for i in range(n_objs)])
        sizes = sizes[sizes > self.volume_threshold]
        sizes_ranks = np.argsort(sizes)[::-1]  # Descending size in class (-1)

        mask = cseg == sizes_ranks[0]
        for i in range(len(sizes_ranks) - 1):
            mask = mask + (cseg == sizes_ranks[i + 1])
        mask = mask != 0
        mask = sitk.BinaryFillhole(mask)
        return sitk.Mask(in_img, mask)


def label_postproc(img_s1: sitk.Image or str,
                   img_s2: sitk.Image or str,
                   out_fname: str = None,
                   skipp_proc: bool = False,
                   ) -> sitk.Image:
    r"""Label post-post processing.
    """
    logger = Logger[mpi.current_process().name]

    # Load image if not already loaded.
    if isinstance(img_s1, str):
        logger.info(f"Reading: {img_s1}")
        img_s1 = sitk.ReadImage(img_s1)
    if isinstance(img_s2, str):
        logger.info(f"Reading: {img_s2}")
        img_s2 = sitk.ReadImage(img_s2)

    # Cast image to uint8, i.e. binary image
    img_s1 = sitk.Cast(img_s1, sitk.sitkUInt8)
    img_s2 = sitk.Cast(img_s2, sitk.sitkUInt8)

    # === Create binary labels for each types of target lesions ===
    air_space = img_s1 == 1
    mt = img_s2 == 1
    mrc = img_s2 == 2

    if not skipp_proc:
        # === Create filters ===
        filter_params = {
            'r_air':        5,
            'r_mt':         2,
            'r_mrc':        3,
            'thres_air':    15, # mm^3
            'thres_mt':     5,
            'thres_mrc':    5
        }
        base_filter = BinaryOpenClose(1, 1, morphological=False) # Removes dots like islands.
        filter_air_space = BinaryOpenClose(filter_params['r_air'])
        filter_mt = BinaryOpenClose(filter_params['r_mt'])
        filter_mrc = BinaryOpenClose(filter_params['r_mrc'])
        volT_air_space = BinaryVolumeThreshold(filter_params['thres_air'])
        volT_mt = BinaryVolumeThreshold(filter_params['thres_mt'])
        volT_mrc = BinaryVolumeThreshold(filter_params['thres_mrc'])

        # === Execute filters ===
        air_space, mt, mrc = [base_filter(x) for x in [air_space, mt, mrc]]
        air_space = volT_air_space(filter_air_space(air_space))
        mt = volT_mt(filter_mt(mt))
        mrc = volT_mrc(filter_mrc(mrc))

    # Value of class in output label
    mapping = {
        'air_space':  1,
        'mt':         2,
        'mrc':        3
    }
    out_im = sitk.Maximum(air_space * mapping['air_space'], mt * mapping['mt'])
    out_im = sitk.Maximum(out_im, mrc * mapping['mrc'])
    if not out_fname is None:
        logger.info(f"Writingg image to: {out_fname}")
        sitk.WriteImage(out_im, out_fname)

    logger.info("Done")
    return out_im

def batch_label_postproc(input_dir_s1: str,
                         input_dir_s2: str,
                         output_dir: str,
                         *args,
                         idglobber: Optional[Pattern] = "^[0-9]+_(L|R)",
                         skip_proc: Optional[bool] = False,
                         **kwargs) -> None:
    # Error check
    assert os.path.isdir(input_dir_s1) and os.path.isdir(input_dir_s1), "Cannot open input directory"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = Logger['msdtk']

    # Grab all files to process
    input_files_s1 = []
    for r, d, f in os.walk(input_dir_s1):
        if len(f) > 0:
            fs = fnmatch.filter(input_files_s1, "*nii*")
            fs = [os.path.join(r, ff) for ff in f]
            input_files_s1.extend(fs)
    input_files_s1.sort()
    s1_ids = get_ids_from_files(input_files_s1, idglobber=idglobber)
    input_files_s2 = get_files_with_id(input_dir_s2, idlist=s1_ids, idglobber=idglobber)

    logger.debug(f"s1_ids: {s1_ids}")
    logger.debug(f"input_files: {input_files_s1}\n{input_files_s2}")

    out_fnames = [f.replace(input_dir_s1, output_dir) for f in input_files_s1]
    # check if directories are there
    for f in out_fnames:
        if not os.path.isdir(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f), exist_ok=True)

    # MPI jobs
    workers = kwargs.get('numworker', 12)
    if workers == 1:
        for _in_file_s1, _in_file_s2, _o_fname in zip(input_files_s1, input_files_s2, out_fnames):
            label_postproc(_in_file_s1, _in_file_s2, _o_fname, skip_proc)
    elif workers >= 2:
        processes = []
        pool = mpi.Pool(kwargs.get('numworker', 12))
        for _in_file_s1, _in_file_s2, _o_fname in zip(input_files_s1, input_files_s2, out_fnames):
            p = pool.apply_async(label_postproc,
                                 args=[_in_file_s1, _in_file_s2, _o_fname, skip_proc])
            processes.append(p)
        pool.close()
        pool.join()


def _call_back_sitk_write(img, fname):
    r"""mpi-call back to write image"""
    logger = Logger[mpi.current_process().name]
    logger.info(f"Writing to: {fname}")
    sitk.WriteImage(img, fname)

# if __name__ == '__main__':
#     batch_label_postproc('../../Sinus/98.Output/v2_B00-v1.0/External/S1_output',
#                          '../../Sinus/98.Output/v2_B00-v1.0/External/S1_output_post',)