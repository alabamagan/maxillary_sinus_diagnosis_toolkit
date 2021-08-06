from pytorch_med_imaging.scripts import remap_label
from pytorch_med_imaging.utils.preprocessing import make_mask_from_dir
from pytorch_med_imaging.logger import Logger
from . import resample_to_standard_spacing
import os
import re
import fnmatch
import multiprocessing as mpi
from typing import AnyStr, List, Union, Pattern, SupportsInt

__all__ = ['pre_processing']

def pre_processing(input_img:   AnyStr,
                   output:      AnyStr,
                   input_gt:    AnyStr=None,
                   idglobber:   Pattern="None",
                   idlist:      AnyStr=None,
                   numworker:   int =10) -> None:
    r"""
    Description
        Preprocessing pipeline that involves three steps:
            1) Normalize image spacing
            2) Create tissue mask using HU value -80, which is rough average of adipose tissues
            3) Convert manual segmentation to: i) air-space only ii) Lesions only iii) Lesions only without
               differentiating types.
        The tissue mask will be generated to `[OUTPUT]/Mask-80` and, if provided, the remapped segmentaitons
        would ge generated to `[OUTPUT]/Seg`.


    Args


    """
    logger = Logger['msdtk']
    #==========================
    # Glob the files to process
    #--------------------------

    # Glob all image files
    img_dirs = recurse_include_files(input_img, idglobber=idglobber)
    if not idlist is None:
        logger.info('Filtering directories')
        if idlist.find(',') >= 0:
            idlist = idlist.split(',') if idlist.find(',') >= 0 else idlist
            idlist = [r.rstrip() for r in idlist]
        else:
            idlist = [idlist]
        img_dirs = {key: img_dirs.get(key, None) for key in idlist}
        if len(gt_dirs) > 0:
            gt_dirs = {key: gt_dirs.get(key, None) for key in idlist}

    # Glob ground-truth labels from directory provided
    gt_dirs = recurse_include_files(input_gt, idglobber=idglobber, idlist=list(img_dirs.keys())) \
        if not input_gt in [None, ""] else {}

    logger.debug(f"{img_dirs}")
    logger.debug(f"{gt_dirs}")

    #==============
    # Normalization
    #--------------
    logger.info("\n{:=^80}".format("Normalization"))
    # Create output file directory
    os.makedirs(output, exist_ok=True)
    pool = mpi.Pool(numworker)

    for key in img_dirs:
        if key in gt_dirs:
            pool.apply(resample_to_standard_spacing,
                             args=[img_dirs[key], output, gt_dirs[key]])
        else:
            pool.apply(resample_to_standard_spacing,
                             args=[img_dirs[key], output])
    pool.close()
    pool.join()


    #=================
    # Make mask for S1
    #-----------------
    logger.info("\n{:=^80}".format('Making mask'))
    mask_dir = os.path.join(output, 'Mask-80')
    os.makedirs(mask_dir, exist_ok=True)

    make_mask_from_dir(output, mask_dir, -80, None, True, num_worker=numworker)

    #=============
    # Remap labels
    #-------------

    if len(gt_dirs) > 0:
        logger.info("{:-^80}".format("Remaping labels"))
        # map original manual segmentation to:
        # 1) air-space only
        # 2) lesion only
        # 3) lesion only (w/o differentiate)
        remap_dict = {
            'Air-space': "{2:1, 3:0, 4:0, 5:0}",
            'Lesion-only': "{1:0, 2:0, 3:1, 4:2, 5:0}",
            'Lesion(single_class)': "{1:0, 2:0, 3:1, 4:1, 5:0}"
        }

        for key in remap_dict:
            remap_output_dir = os.path.join(output, 'Seg', key)
            os.makedirs(remap_output_dir, exist_ok=True)
            a = ['-i', os.path.join(output, 'Seg'), '-o', remap_output_dir, '-n', str(numworker),
                 '-m', remap_dict[key]]
            logger.info(f'Passing arguments: {a}')
            remap_label(a)
    else:
        logger.info("Skipping label remap because no manual segmentation is provided.")
    logger.info("{:=^80}".format("Done"))


def recurse_include_files(input_img, idglobber=None, idlist=None):
    logger = Logger['recurse_include_files']
    img_dirs = {}
    for r, d, f in os.walk(input_img):
        if not len(f) == 0:
            f = fnmatch.filter(f, '*.nii.gz')
            if not idglobber is None:
                ids = [re.search(idglobber, ff).group() for ff in f]
            else:
                ids = [ff.rstrip('.nii.gz') for ff in f]

            for id, fname in zip(ids, f):
                if id in img_dirs:
                    logger.warning(f"The id '{id}' is not unique! {os.path.join(r,fname)} repeats with {img_dirs[id]}",
                                   True)
                    logger.warning(f"Replacing {id}")
                img_dirs[id] = os.path.join(r, fname)

    if not idlist is None:
        if isinstance(idlist, str):
            idlist = idlist.split(',')

        poping = []
        for key in img_dirs:
            if not key in idlist:
                poping.append(key)
        [img_dirs.pop(key) for key in poping]
    return img_dirs


