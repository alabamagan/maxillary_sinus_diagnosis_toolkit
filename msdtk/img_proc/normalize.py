import torchio as tio
import SimpleITK as sitk
import nibabel
import os
import numpy as np
import multiprocessing as mpi
import torchio as tio
from pathlib import Path

from pytorch_med_imaging.logger import Logger
from typing import Union, Callable, Iterable, Optional

__all__ = ['resample_to_standard_spacing', 'intensity_normalization_inference', 'intensity_normalization_train']

def resample_to_standard_spacing(in_fname: str,
                                 out_dir: str,
                                 seg_fname: Optional[str] = None) -> None:
    r"""
    Resample the input NIFTI file to standard spacing (0.4 mm)

    Args:
        in_fname (str):
            File path of the input. Must be nifti.
        out_dir (str):
            Directory to dump the output. Will be created if not exist.
        seg_fname (str, Optional):
            File path of the corresponding segmentation. Must be nifti.

    """
    assert os.path.isfile(in_fname), "Cannot open input."
    assert in_fname.endswith('.nii.gz') or in_fname.endswith('.nii'), "Only accepts NIFTI."

    logger = Logger['msdtk']

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    target_spacing = np.asarray([.4, .4, .4])

    # Use nibabel to check image and save load time
    im = nibabel.load(in_fname)
    im_header = im.header
    im_header = {key: im_header.structarr[key].tolist() for key in im_header.structarr.dtype.names}
    spacing = [float(im_header['pixdim'][j]) for j in range(1, 4)]

    # Define output directories.
    out_fname = os.path.join(out_dir, os.path.basename(in_fname))
    if not all(np.isclose(spacing, [0.4, 0.4, 0.4])):    # resample if not same spacing
        sitk_im = sitk.ReadImage(in_fname)

        # Compute new size
        target_size = np.asarray(sitk_im.GetSize()) * np.asarray(sitk_im.GetSpacing()) / target_spacing
        target_size = target_size.astype('int').tolist()
        logger.info(f"Resampling: {in_fname}")
        logger.debug(f"Source size -> Target size: {list(sitk_im.GetSize())} {target_size}")

        # resample
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetReferenceImage(sitk_im)
        resample_filter.SetOutputSpacing(target_spacing)
        resample_filter.SetSize(target_size)
        resample_filter.SetInterpolator(sitk.sitkBSpline)
        out_im = resample_filter.Execute(sitk_im)
        sitk.WriteImage(out_im, out_fname)

        if not seg_fname is None:
            out_segfname = os.path.join(out_dir, 'Seg', os.path.basename(seg_fname))
            os.makedirs(os.path.join(out_dir, 'Seg'), exist_ok=True)
            # Set to nearest neighbour first
            resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
            out_seg = resample_filter.Execute(sitk.ReadImage(seg_fname))
            sitk.WriteImage(out_seg, out_segfname)
    else:
        logger.info(f"Nothing is done to {in_fname}{' and segmentation not provided' if seg_fname is None else seg_fname}.")
        try:
            os.symlink(os.path.relpath(in_fname, out_dir), out_fname)
        except OSError:
            os.remove(out_fname)
            os.symlink(os.path.relpath(in_fname, out_dir), out_fname)
        except:
            logger.warning(f"{in_fname} was not linked.")
        if not seg_fname is None:
            seg_path = os.path.join(out_dir, 'Seg')
            out_segfname = os.path.join(out_dir, 'Seg', os.path.basename(seg_fname))
            os.makedirs(os.path.join(out_dir, 'Seg'), exist_ok=True)
            try:
                os.symlink(os.path.relpath(seg_fname, seg_path), out_segfname)
            except OSError:
                os.remove(out_segfname)
                os.symlink(os.path.relpath(seg_fname, seg_path), out_segfname)
            except:
                logger.warning(f"{seg_fname} was not linked.")


def intensity_normalization_train(imgs_dir: Union[str, Path, Iterable[Union[str, Path]]],
                                  out_landmark_path: Union[str, Path],
                                  mask: Optional[Union[str, Iterable[Union[str, Path]], Callable]] = None,
                                  recurisve_include: Optional[bool] = False) -> None:
    r"""
    Use the torchio API to perform histogram normalization
    """
    import fnmatch

    out_landmark_path = Path(out_landmark_path)
    if not out_landmark_path.parent.is_dir():
        out_landmark_path.parent.mkdir(exist_ok=True)

    if recurisve_include:
        imgs = []
        for r, d, f in os.listdir(imgs_dir):
            if len(f):
                f = fnmatch.filter(f, "*.nii.gz")
                imgs.extend([os.path.join(r, ff) for ff in f])
    else:
        imgs = os.listdir(imgs_dir.absolute())

    masking_function = mask if callable(mask) else None
    masking_path = mask if isinstance(mask, [list, tuple]) else None
    if not masking_path is None:
        if len(masking_path) != len(imgs):
            raise IndexError("Length of images and mask are different")
    if mask == 'corner': # Use corner pixel as background
        masking_function = lambda x: x > x.flatten()[0]


    hist_norm = tio.transforms.HistogramStandardization.train(imgs, masking_function=masking_function, masking_path=masking_path)
    np.save(out_landmark_path)
    pass

def intensity_normalization_inference(imgs_dir: Union[str, Path, Iterable[Union[str, Path]]],
                                      output_dir: Union[str, Path],
                                      landmark_path: Union[str, Path],
                                      mask: Optional[Union[str, Callable, None]] = None,
                                      recursive_include: Optional[bool] = False,
                                      numworker: Optional[int] = 16) -> None:
    r"""
    Perform normalization
    """
    import fnmatch
    from functools import partial

    # Error check
    landmark_path = Path(landmark_path)
    assert Path.is_file(landmark_path), f"Cannot onpen landmark at {landmark_path}"
    if isinstance(output_dir, str): # Make sure this is a directory
        if not output_dir.endswith(os.path.sep):
            output_dir += os.path.sep


    logger = Logger['intensity_normalization']

    if recursive_include:
        imgs = []
        for r, d, f in os.listdir(imgs_dir):
            if len(f):
                f = fnmatch.filter(f, "*.nii.gz")
                imgs.extend([os.path.join(r, ff) for ff in f])
    else:
        imgs = os.listdir(Path(imgs_dir).absolute())
        imgs = fnmatch.filter(imgs, '*.nii.gz')
        imgs = [os.path.join(imgs_dir, ff) for ff in imgs]


    logger.info(f"Found {len(imgs)} files: \n{imgs}")

    logger.info(f"Loading landmarks from: {landmark_path}")
    landmarks = {'image': np.load(landmark_path)}

    if mask == 'corner': # Use corner pixel as background
        masking_function = _corner_masking
    else:
        masking_funciton = None



    logger.info("Performing transform...")
    if numworker == 1:
        for f in imgs:
            _normalization_transform(f, landmarks, imgs_dir, output_dir, masking_function)
    elif numworker > 1:
        logger.info("MPI mode")
        process = []
        pool = mpi.Pool(numworker)
        for f in imgs:
            p = pool.apply_async(_normalization_transform,
                                 args=[f, landmarks, imgs_dir, output_dir, masking_function])
            process.append(p)
        pool.close()
        pool.join()


def _normalization_transform(f, landmarks, imgs_dir, output_dir, masking_function):
    r"""MPI Wrapping function"""
    logger = Logger['msdtk-'+mpi.current_process().name]
    hist_norm = tio.transforms.HistogramStandardization(landmarks, masking_method=masking_function)

    logger.info(f"Processing: {f}")
    s = tio.Subject(image=tio.ScalarImage(f))
    s_normed = hist_norm(s)
    out_name = Path(f.replace(imgs_dir, output_dir).replace('.nii.gz', '_normed.nii.gz'))
    if not out_name.parent.is_dir():
        out_name.parent.mkdir(exist_ok=True)
    logger.info(f"Writing to: {out_name}")
    s_normed.image.save(out_name)

def _corner_masking(x):
    return x > x.flatten()[0]