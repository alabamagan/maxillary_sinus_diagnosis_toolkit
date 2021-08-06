import torchio as tio
import SimpleITK as sitk
import nibabel
import os
import numpy as np
from pytorch_med_imaging.logger import Logger

__all__ = ['resample_to_standard_spacing']

def resample_to_standard_spacing(in_fname: str,
                                 out_dir: str,
                                 seg_fname: str = None) -> None:
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


def intensity_normalization():
    r"""
    Use the torchio API to perform histogram normalization
    """
    import torchio as tio

    pass