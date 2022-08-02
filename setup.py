#!/usr/bin/env python

import sys, os
from glob import glob

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from distutils.command.install_headers import install_headers
from setuptools import find_packages
from setuptools import setup

NAME =               'neurokernel'
VERSION =            '0.3.1'
AUTHOR =             'Neurokernel Development Team'
AUTHOR_EMAIL =       'neurokernel-dev@columbia.edu'
URL =                'https://github.com/neurokernel/neurokernel/'
MAINTAINER =         AUTHOR
MAINTAINER_EMAIL =   AUTHOR_EMAIL
DESCRIPTION =        'An open architecture for Drosophila brain emulation'
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']

# Explicitly switch to parent directory of setup.py in case it
# is run from elsewhere:
os.chdir(os.path.dirname(os.path.realpath(__file__)))
PACKAGES =           find_packages()

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        packages = PACKAGES,
        include_package_data = True,
        install_requires = [
            'bidict >= 0.11.0',
            'dill >= 0.2.4, <= 0.3.3',
            'future >= 0.16.0',
            'h5py >= 2.8.0',
            'lxml >= 3.3.0',
            'markupsafe >= 0.23',
            'matplotlib >= 1.4.3',
            'mpi4py >= 1.3.1',
            'networkx >= 2.4',
            'numexpr >= 2.3',
            'numpy >= 1.9.2',
            'pandas >= 1.0.0',
            'ply >= 3.4',
            'psutil >= 2.2.1',
            'pycuda >= 2020.1',
            'scipy >= 0.11.0',
            'shutilwhich >= 1.1.0',
            'twiggy >= 0.4.7',
            'tqdm'
        ],
        extras_require = {
            'sphinx': ['sphinx >= 1.3'],
            'sphinx_rtd_theme': ['sphinx_rtd_theme >= 0.1.6'],
            ':python_version == "2.7"': ['futures'],
            }
        )
