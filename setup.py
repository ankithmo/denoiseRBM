
from setuptools import setup, find_packages
from os import path as osp

import sys
from io import open

here = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(here, 'dRBM'))
from version import __version__

# Get the long description from the README file
with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='denoiseRBM',
      version=__version__,
      description='Alleviating the noisy data problem using restricted Boltzmann machines',
      url='https://github.com/ankithmo/denoiseRBM',
      author='Ankith Mohan',
      author_email='ankithmo@usc.edu',
      keywords=['pytorch', 'graph machine learning', 'graph representation learning', 'graph neural networks', 'restricted Boltzmann machines', 'noisy data problem', 'graph denoising'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires = [
        'torch>=1.2.0',
        'numpy>=1.16.0',
        'tqdm>=4.29.0',
        'scikit-learn>=0.20.0',
        'pandas>=0.24.0',
        'six>=1.12.0',
        'urllib3>=1.24.0',
        'ogb==1.1.1',
        'torch_geometric==1.5.0'
      ],
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