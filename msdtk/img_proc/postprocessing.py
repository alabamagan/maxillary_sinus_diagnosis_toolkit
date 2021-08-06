import os
import re, fnmatch
from tqdm.auto import tqdm
import SimpleITK as sitk
import multiprocessing as mpi
from pytorch_med_imaging.logger import Logger
from functools import partial

__all__ = ['label_postproc', 'batch_label_postproc']


def label_postproc(img: sitk.Image or str,
                   out_fname: str = None,
                   opening_radius: int = 5,
                   closing_radius: int = 5,
                   connected_component: int = -1) -> sitk.Image:
    r"""

    Args:
        img:

    Returns:
        sitk.Image
    """
    logger = Logger[mpi.current_process().name]

    # Load image if not already loaded.
    if isinstance(img, str):
        logger.info(f"Processing: {img}")
        img = sitk.ReadImage(img)

    # Cast image to uint8, i.e. binary image
    img = sitk.Cast(img, sitk.sitkUInt8)

    # === Create filter ===
    # 1) opening (remove small objects)
    # 2) cloasing (remove small holes)
    # 3) Fill holes
    # 4) Keeping only two largest connected components.

    opening_filter = sitk.BinaryOpeningByReconstructionImageFilter()
    opening_filter.SetKernelRadius(opening_radius)
    opening_filter.SetKernelType(sitk.sitkBall)

    closing_filter = sitk.BinaryClosingByReconstructionImageFilter()
    closing_filter.SetKernelRadius(closing_radius)
    closing_filter.SetKernelType(sitk.sitkBall)

    fillhole_filter = sitk.BinaryFillholeImageFilter()
    fillhole_filter.SetFullyConnected(False)


    # === Execute filters ===
    out = fillhole_filter.Execute(closing_filter.Execute(opening_filter.Execute(img)))
    out.CopyInformation(img)

    # Extract largest connected components
    if connected_component > 0:
        logger.info("Extracting connected components.")

        binary_out = sitk.BinaryThreshold(out, 1, 255)

        cc_filter = sitk.ConnectedComponentImageFilter()
        cseg = cc_filter.Execute(binary_out)
        n_objs = cc_filter.GetObjectCount()

        if n_objs < 2:
            raise ArithmeticError("Number of segmented objects less than 2!")

        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(cseg)

        sizes = [shape_stats.GetPhysicalSize(i + 1) for i in range(n_objs)]
        sizes_ranks = np.argsort(sizes)[::-1]  # Descending size in class (-1)

        mask = cseg == sizes_ranks[0]
        for i in range(connected_component):
            mask = mask + (cseg == sizes_ranks[i + 1])

        out = sitk.Mask(out, mask)

    if not out_fname is None:
        logger.info(f"Writingg image to: {out_fname}")
        sitk.WriteImage(out, out_fname)

    logger.info("Done")
    return out

def batch_label_postproc(input_dir: str,
                         output_dir: str,
                         connected_components: int,
                         kernel_radius: int,
                         *args, **kwargs) -> None:
    # Error check
    assert os.path.isdir(input_dir), "Cannot open input directory"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Grab all files to process
    input_files = []
    for r, d, f in os.walk(input_dir):
        if len(f) > 0:
            fs = fnmatch.filter(input_files, "*nii*")
            fs = [os.path.join(r, ff) for ff in f]
            input_files.extend(fs)
    input_files.sort()
    out_fnames = [f.replace(input_dir, output_dir) for f in input_files]
    # check if directories are there
    for f in out_fnames:
        if not os.path.isdir(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f), exist_ok=True)

    # MPI jobs
    processes = []
    pool = mpi.Pool(kwargs.get('numworker', 12))
    for _in_file, _o_fname in zip(input_files, out_fnames):
        p = pool.apply_async(partial(label_postproc,
                                     opening_radius=kernel_radius,
                                     closing_radius=kernel_radius,
                                     connected_component=connected_components),
                             args=[_in_file, _o_fname])
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