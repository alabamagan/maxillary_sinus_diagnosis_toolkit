
from .console_entry import ConsoleEntry as ArgumentParser # This rename helps guild to parse this script
from ..img_proc import batch_label_postproc
import argparse
import sys

__all__ = ['post_proc_console_entry']

def post_proc_console_entry(*args, **kwargs):
    parser = ArgumentParser('onLvg',
                            description="Pre-proccessing to align inputs for msdtk standard."
                                        "This includes normalizing the spacing, creating mask"
                                        "for step 1 and remap labels for step 2 if ground-truth"
                                        "are provided. Which will be generated to OUTPUT/Seg")
    parser.add_argument('-i', '--input-img', action='store',
                        help='Directory to images')
    parser.add_argument('-c', '--connected-components', action='store', type=int, default=-1,
                        help='If positive output labels will only keep up to X connected components.')
    parser.add_argument('-r', '--kernel-radius', action='store', type=int, default=5,
                        help='Radius of the kernels in opening/closing operation.')
    a = parser.parse_args(*args, **kwargs)
    parser.logger.info(f"{a}")

    batch_label_postproc(a.input_img, a.output, a.connected_components, a.kernel_radius, numworker=a.numworker)

if __name__ == '__main__':
    post_proc_console_entry(sys.argv[1:])