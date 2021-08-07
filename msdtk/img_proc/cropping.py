import SimpleITK as sitk
import numpy as np
import os
import multiprocessing as mpi

from pathlib import Path
from typing import Union, Iterable, List, Pattern

from ast import literal_eval as eval
from typing import *
from pytorch_med_imaging.med_img_dataset import ImageDataSet
from pytorch_med_imaging.logger import Logger

from ..utils import get_ids_from_files


__all__ = ['batch_crop_sinuses', 'crop_sinuses']

def batch_crop_sinuses(dir_pairs: List[Tuple[str, str]],
                       idglobber: Pattern = "^[0-9]+",
                       num_workers: int = 8,
                       load_bounds: str = None,
                       save_bounds: bool = False,
                       skip_exist: bool = False):
    r"""
    This function can be called to crop the sinuses from the CBCT images using their segmentation.
    This assumes the two sinus (at list the air space within) are well segmented and each input image
    has two sinuses.

    The boundaries computed from the first directory pair are reused to crop all the images in the
    directory pairs.

    The directory pairs should be arranged as follow:

        `dir_paris=[(input_dir_1, output_dir_1),(input_dir_2, output_dir_2), ...]`

    At least one pair should be given.

    Args:
        dir_pairs (list of tuples):
            Directory pairs, structured like: [(input1, output1), (input2, output2), ...]
        bounds (str or list):
            Directory of csv that hold all the boundaries to crop.
    """
    assert isinstance(dir_pairs, list)

    logger = Logger['msdtk']

    for indir, outdir in dir_pairs:
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

    # First pair is the referenced pair, must be image labels
    first_in_dir, first_out_dir = dir_pairs[0]
    first_set = ImageDataSet(first_in_dir, verbose=True, dtype='uint8', idGlobber=idglobber, debugmode=False)
    idlist = first_set.get_unique_IDs()

    logger.debug(f"Globbed ids: {idlist}")

    # Glob ids existing in the output directory and remove it from the idlist
    if skip_exist:
        logger.info(f"Skip existing.")
        existing_ids = get_ids_from_files(first_out_dir, return_dict=False)
        for _id in existing_ids:
            idlist.remove(_id)
        if len(idlist) == 0:
            logger.warning("Nothing left to crop!")
            return 0

    if load_bounds is None:
        res = find_sinus_bounds(first_set)
    else:
        import pandas as pd
        assert os.path.isfile(load_bounds)
        res = pd.read_csv(load_bounds, index_col=0)
        res.index = res.index.astype("str")
        res = res.loc[idlist]
        res = [[eval(str(right)), eval(str(left))] for right, left in res[['right', 'left']].to_numpy()]

        # Also write the indexes
    if save_bounds and load_bounds is None:
        import pandas as pd
        right, left = [list(a) for a in zip(*res)]
        df = pd.DataFrame.from_dict({'right': right, 'left': left})
        df.index = idlist
        df.index.name = "ID"
        df.to_csv(os.path.join(first_out_dir, 'cropping_index.csv'))

    crop_images_with_bounds(dir_pairs, first_set, idlist, res)
    return 0


def get_LR_axis(direction_mat: np.ndarray) -> int:
    r"""Determine which column vector in the direction_mat is closest to x-direction"""
    if not isinstance(direction_mat, np.ndarray):
        direction_mat = np.asarray(direction_mat)
    direction_mat = direction_mat.reshape(3, 3)

    x_vect = np.asarray([1, 0, 0])
    dp = [np.abs(x_vect.dot(col_vect)) for col_vect in direction_mat.T]
    lr_axis = np.argmax(dp)
    lr_vect = direction_mat[:, lr_axis]
    return lr_axis, lr_vect


def crop_sinuses(seg: sitk.Image or str,
                 return_bounds: bool = False) -> List[sitk.Image] or List[List[int]]:
    r"""Crop `img` at the two sinux and return the cropped segmentation."""
    logger = Logger[mpi.current_process().name]
    logger.info("Processing")

    if isinstance(seg, str):
        logger.info(f"Reading {seg}")
        segname = seg
        seg = sitk.ReadImage(seg)
    else:
        segname = None



    # create a binary version of segmentation
    bseg = sitk.BinaryThreshold(seg, lowerThreshold=1, upperThreshold=65535)
    bseg = sitk.BinaryMorphologicalClosing(bseg, 3, sitk.sitkBall)

    # Get two largest connected compoenent
    cc_filter = sitk.ConnectedComponentImageFilter()
    cseg = cc_filter.Execute(bseg)
    n_objs = cc_filter.GetObjectCount()

    if n_objs < 2:
        raise ArithmeticError("Number of objects less than 2 for image: {}".format(segname))


    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(cseg)

    sizes = [shape_stats.GetPhysicalSize(i + 1) for i in range(n_objs)]
    sizes_ranks = np.argsort(sizes)[::-1] # Descending size in class (-1)

    sorted_cent = [shape_stats.GetCentroid(int(i + 1)) for i in sizes_ranks]
    sorted_bb = [shape_stats.GetBoundingBox(int(i + 1)) for i in sizes_ranks]
    sorted_bb_ini = [bb[:3] for bb in sorted_bb]
    sorted_bb_size = [bb[3:] for bb in sorted_bb]
    sorted_bb_fin = [[a + b  for a, b in zip(bbi, bbs)] for bbi, bbs in zip(sorted_bb_ini, sorted_bb_size)]

    # Get the largest two labels
    cseg_1 = seg[sorted_bb_ini[0][0]:sorted_bb_fin[0][0],
                 sorted_bb_ini[0][1]:sorted_bb_fin[0][1],
                 sorted_bb_ini[0][2]:sorted_bb_fin[0][2]]
    cseg_2 = seg[sorted_bb_ini[1][0]:sorted_bb_fin[1][0],
                 sorted_bb_ini[1][1]:sorted_bb_fin[1][1],
                 sorted_bb_ini[1][2]:sorted_bb_fin[1][2]]

    # Get LR axis
    LR_axis, LR_vector = get_LR_axis(seg.GetDirection())
    leftness = [np.asarray(a).dot(LR_vector) for a in sorted_cent] # Larger = left side
    leftlabel_is_0th = leftness[0] > leftness[1]

    # Return [Right, Left]
    if  not leftlabel_is_0th:
        out = [cseg_1, cseg_2]
        if return_bounds:
            out = [tuple(sorted_bb_ini[0]) + tuple(sorted_bb_fin[0]),
                   tuple(sorted_bb_ini[1]) + tuple(sorted_bb_fin[1])]
    else:
        out = [cseg_2, cseg_1]
        if return_bounds:
            out = [tuple(sorted_bb_ini[1]) + tuple(sorted_bb_fin[1]),
                   tuple(sorted_bb_ini[0]) + tuple(sorted_bb_fin[0])]
    return out


def crop_images_with_bounds(dir_pairs: List[Tuple[int]],
                            first_set: ImageDataSet,
                            idlist: List[str],
                            res: List[List[int]]):
    r"""Crop the images according to the bounds `res`"""
    import tqdm.auto as auto

    for indir, outdir in dir_pairs:
        in_seg = ImageDataSet(indir, verbose=True, filtermode='idlist', idlist=idlist,
                              idGlobber=first_set._id_globber)
        for j, seg_path in enumerate(auto.tqdm(in_seg.data_source_path)):
            right, left = res[j]
            img = sitk.ReadImage(seg_path)

            img_right = img[right[0]:right[3],
                        right[1]:right[4],
                        right[2]:right[5]]
            img_left = img[left[0]:left[3],
                       left[1]:left[4],
                       left[2]:left[5]]

            sitk.WriteImage(img_right, os.path.join(outdir, idlist[j] + '_R.nii.gz'))
            sitk.WriteImage(img_left, os.path.join(outdir, idlist[j] + '_L.nii.gz'))

            del img_left, img_right, img


def find_sinus_bounds(input_set: ImageDataSet,
                      num_worker: int = 8) -> List[List[int]]:
    r"""
    Find the bound of the sinus from the inputs. Input `first_set` must be a :class:`ImagedataSet` and
    must be integer datatype. This method is not thread safe and should not be called from a thread.

    The output are in formats of:
        [(right_sinus_bounds, left_sinus_bounds), ...]
    """
    from functools import partial

    logger = Logger['msdtk']

    idlist = input_set.get_unique_IDs()
    pool = mpi.Pool(num_worker)

    logger.info("Creating jobs.")
    res = pool.map_async(partial(crop_sinuses, return_bounds=True), input_set.data_source_path)
    res = res.get()
    pool.close()
    pool.join()
    return res

