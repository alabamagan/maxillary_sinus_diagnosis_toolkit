import os
import argparse
from configparser import ConfigParser

def update_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', action='store')
    parser.add_argument('--target-dir', action='store')
    parser.add_argument('--validation-dir', action='store')
    parser.add_argument('--fold-code', action='store')
    a = parser.parse_args()

    script_path = os.path.dirname(os.path.realpath(__file__))
    s1_config = ConfigParser.read(os.path.join(script_path, '../../pmi_config/Seg_S1.ini'))
    s2_config = ConfigParser.read(os.path.join(script_path, '../../pmi_config/Seg_S2.ini'))

    s1_config['Data']['input_dir'] = a.input_dir
    s1_config['Data']['target_dir'] = a.target_dir
    s1_config['Data']['validation_dir'] = a.validation_dir
    s1_config['General']['fold_code'] = a.fold_code

    s2_config['Data']['input_dir'] = a.input_dir
    s2_config['Data']['target_dir'] = a.target_dir
    s2_config['Data']['validation_dir'] = a.validation_dir
    s2_config['General']['fold_code'] = a.fold_code