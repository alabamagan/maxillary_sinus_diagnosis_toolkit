import argparse
import os
from pytorch_med_imaging.logger import Logger

__all__ = ['ConsoleEntry']

class ConsoleEntry(argparse.ArgumentParser):
    r"""
    This class is a wrapped parser which contains some presets:

    +-------+------+-------------+------------+---------+-----------------------+
    | label | flag | name        | action     | default | description           |
    +=======+======+=============+============+=========+=======================+
    | i     | -i   | --input     | store      | N/A     | Input file directory  |
    +-------+------+-------------+------------+---------+-----------------------+
    | o     | -o   | --output    | store      | N/A     | Output directory name |
    +-------+------+-------------+------------+---------+-----------------------+
    | O     | -o   | --outfile   | store      | N/A     | Output file name      |
    +-------+------+-------------+------------+---------+-----------------------+
    | g     | -g   | --idglobber | store      | None    | Regex for ID globbing |
    +-------+------+-------------+------------+---------+-----------------------+
    | L     | -l   | --idlist    | store      | None    | List or text file     |
    +-------+------+-------------+------------+---------+-----------------------+
    | n     | -n   | --numworker | store      | 10      | Number of workers     |
    +-------+------+-------------+------------+---------+-----------------------+
    | v     | -v   | --verbose   | store_true | False   | Verbosity setting     |
    +-------+------+-------------+------------+---------+-----------------------+

    Attribute:
        logger (msdtk.Logger):
            Use this for logging.

    """
    def __init__(self, addargs: str, *args, **kwargs):
        super(ConsoleEntry, self).__init__(*args, **kwargs)

        default_arguments = {
            'i': (['-i', '--input'],    {'type': str, 'help': 'Input directory that contains nii.gz or DICOM files.'}),
            'o': (['-o', '--output'],   {'type': str, 'help': 'Directory for generated output.'}),
            'O': (['-o', '--outfile'],  {'type': str, 'help': 'Directory for generated file.'}),
            'g': (['-g', '--idglobber'],{'type': str, 'help': 'Globber for globbing case IDs.', 'default': None}),
            'L': (['-l', '--idlist'],   {'type': str, 'help': 'List or txt file directory for loading only specific ids.', 'default': None}),
            'n': (['-n', '--numworker'],{'type': int, 'help': 'Specify number of workers.', 'default': 10}),
            'v': (['-v', '--verbose'],  {'action': 'store_true', 'help': 'Verbosity.'}),
        }

        for k in addargs:
            args, kwargs = default_arguments[k]
            self.add_argument(*args, **kwargs)

        # For convinient, but not very logical to put this here
        self.logger = Logger('msdtk.log', logger_name='msdtk', verbose=False, keep_file=False)

    @staticmethod
    def make_console_entry_io():
        return pmi_console_entry('iogLv')


    def parse_args(self, *args, **kwargs):
        a = super(ConsoleEntry, self).parse_args(*args, **kwargs)
        if hasattr(a, 'verbose'):
            self.logger.set_verbose(a.verbose)
            self.logger.info(f"Recieved arguments: {a}")

        # Create output dir
        if hasattr(a, 'output'):
            if not os.path.isdir(a.output):
                os.makedirs(a.output, exist_ok=True)
        return a

class readable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values

        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentError("readable_dir:{0} is not a valid path".format(prospective_dir))

        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentError("readable_dir:{0} is not a readable dir".format(prospective_dir))
