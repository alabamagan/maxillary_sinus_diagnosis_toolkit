# python setup.py build_ext --inplace

from setuptools import setup
from setuptools.extension import Extension
import numpy
import os

scripts = [os.path.join('msdtk/scripts', s) for s in os.listdir('msdtk/scripts')]

setup(
    name='Maxillary_Sinus_Diagnosis_Toolkit',
    version='0.1',
    packages=['msdtk'],
    url='https://github.com/alabamagan/maxillary_sinus_diagnosis_toolkit',
    license='',
    author='MLW',
    author_email='fromosia@link.cuhk.edu.hk',
    description='',
    entry_points = {
        'console_scripts': [
            'msdtk-label_statistics = msdtk.scripts:msdtk_label_statistics',
            'msdtk-crop_sinuses = msdtk.scripts:crop_sinuses',
            'msdtk-seg2diag_train = msdtk.scripts:train_seg2diag',
            'msdtk-seg2diag_inference = msdtk.scripts:inference_seg2diag',
            'msdtk-pre_processing = msdtk.scripts.preprocessing:pre_proc_console_entry',
            'msdtk-post_processing = msdtk.scripts.postprocessing:post_proc_console_entry',
        ]
    },
    # scripts = scripts,
    install_requires=['pytorch_medical_imaging'],
    # dependency_links=[os.path.abspath('./ThirdParty/torchio')]
)
