
from setuptools import setup, find_packages
from os import path as osp

import sys
from io import open

here = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(here, 'denoiseRBM'))
from version import __version__

# Get the long description from the README file
with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='denoiseRBM',
      version=__version__,
      description='Alleviating the noisy data problem in graph-structured datasets using restricted Boltzmann machines',
      url='https://github.com/ankithmo/denoiseRBM',
      author='Ankith Mohan',
      author_email='ankithmo@usc.edu',
      keywords=['pytorch', 'graph machine learning', 'graph representation learning', 'graph neural networks', 'restricted Boltzmann machines', 'noisy data problem', 'graph denoising'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT',
      include_package_data=True,
      classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
    ],
)