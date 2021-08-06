

from .console_entry import ConsoleEntry as ArgumentParser # This rename helps guild to parse this script
from ..img_proc import pre_processing
import argparse
import sys
__all__ = ['pre_proc_console_entry']

def pre_proc_console_entry(*args, **kwargs):
    parser = ArgumentParser('onLvg',
                            description="Pre-proccessing to align inputs for msdtk standard."
                                        "This includes normalizing the spacing, creating mask"
                                        "for step 1 and remap labels for step 2 if ground-truth"
                                        "are provided. Which will be generated to OUTPUT/Seg.")
    parser.add_argument('-i', '--input-img', action='store',
                        help='Directory to images')
    parser.add_argument('-gt', '--input-gt', action='store', default=None,
                        help="Optional. Directory to ground-truth manual segmentation. The ground-truth will also be "
                             "used for normalization and output to the specified locations.")
    a = parser.parse_args(*args, **kwargs)
    parser.logger.info(f"{a}")

    pre_processing(a.input_img, a.output, a.input_gt, a.idglobber, a.idlist, a.numworker)
    
if __name__ == '__main__':
    pre_proc_console_entry(sys.argv[1:])