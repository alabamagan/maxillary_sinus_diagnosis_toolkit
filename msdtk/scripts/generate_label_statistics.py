from ..img_proc import label_statistics
from .console_entry import ConsoleEntry


__all__ = ['msdtk_label_statistics']

def msdtk_label_statistics():
    r"""
    Compute the volume, roundness and curvatures of the segmentation.
    """
    parser = ConsoleEntry('iOgLnv')
    parser.add_argument('--normalize', action='store_true', help="Normalize the pixel count.")
    args = parser.parse_args()

    df = label_statistics(args.input, args.idglobber, args.numworker, verbose=args.verbose, normalized=args.normalize)
    if args.outfile.endswith('.csv'):
        df.to_csv(args.outfile)
    else:
        df.to_excel(args.outfile)

