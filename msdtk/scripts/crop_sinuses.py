from .console_entry import ConsoleEntry
from ..img_proc import batch_crop_sinuses

__all__ = ['crop_sinuses']

def crop_sinuses():
    r"""
    Crop the image into left and right sinus based on segmentation.
    """
    parser = ConsoleEntry('Lgnv')
    parser.add_argument('-i', '--input-dirs', action='append', dest='input_dirs',
                        help="Specify the input directory that contains the images (nii.gz). The first"
                             "should always be label images.")
    parser.add_argument('-o', '--output-dirs', action='append', dest='output_dirs',
                        help="Sepcify the otuput directory where the cropped images will be (nii.gz).")
    parser.add_argument('--load-bounds', action='store', dest='load', type=str, default=None,
                        help="If provided, images are cropped using the provided bounds. "
                             "Cannot be used with --save-bounds")
    parser.add_argument('--save-bounds', action='store_true', dest='save',
                        help="If true, bounds of the input computed will be saved as a text file "
                             "in the same directory as the outputs with filename bounds.txt")
    args = parser.parse_args()

    assert len(args.input_dirs) == len(args.output_dirs), "Different number of input and output directories."

    dir_pairs = [(a, b) for a, b in zip(args.input_dirs, args.output_dirs)]
    batch_crop_sinuses(dir_pairs,
                       num_workers=args.numworker,
                       load_bounds=args.load,
                       save_bounds=args.save)

