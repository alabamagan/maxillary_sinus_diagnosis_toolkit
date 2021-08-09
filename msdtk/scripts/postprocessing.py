
from .console_entry import ConsoleEntry as ArgumentParser # This rename helps guild to parse this script
from ..img_proc import batch_label_postproc
import argparse
import sys

__all__ = ['post_proc_console_entry']

def post_proc_console_entry(*args, **kwargs):
    parser = ArgumentParser('onvg',
                            description="Pre-proccessing to align inputs for msdtk standard."
                                        "This includes normalizing the spacing, creating mask"
                                        "for step 1 and remap labels for step 2 if ground-truth"
                                        "are provided. Which will be generated to OUTPUT/Seg")
    parser.add_argument('-s1', '--input-img-s1', action='store',
                        help='Directory to images')
    parser.add_argument('-s2', '--input-img-s2', action='store',
                        help='Directory to images')
    parser.add_argument('--skip-proc', action='store_true',
                        help='Skip the processing, only add the labels together.')
    a = parser.parse_args(*args, **kwargs)
    parser.logger.info(f"{a}")

    batch_label_postproc(a.input_img_s1, a.input_img_s2, a.output, numworker=a.numworker,
                         skip_proc=a.skip_proc)

if __name__ == '__main__':
    post_proc_console_entry(sys.argv[1:])