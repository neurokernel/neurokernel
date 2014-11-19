.. -*- rst -*-

Installation
============

Quick Start
-----------
Neurokernel requires at least one NVIDIA GPU and `CUDA 
<http://www.nvidia.com/object/cuda_home_new.html>`_.

Make sure you have `pip <http://pip.pypa.io>`_ installed (preferably
in a `virtualenv <http://virtualenv.pypa.io>`_); once you do, install the
following dependencies as follows::

  pip install numpy
  pip install cython
  pip install numexpr
  pip install tables

You will also need to have `ffmpeg <http://www.fmpeg.org>`_ or `libav 
<http://libav.org>`_ installed to generate some of the demo visualizations.

Run the following to install the remaining dependencies and the latest 
Neurokernel code::

  git clone https://github.com/neurokernel/neurokernel.git
  pip install -e git+./neurokernel#egg=neurokernel

Supported Platforms
-------------------
Neurokernel has been tested and installed on Linux. It may run on other
platforms too; if you encounter problems, submit a bug report on
`GitHub <https://github.com/neurokernel/neurokernel/issues>`_.

Installation Dependencies
-------------------------
In addition to Python 2.7 and NVIDIA CUDA, Neurokernel currently requires the
following packages:

* `bidict <http://pypi.python.org/pypi/bidict/>`_ 0.1.0 or later.
* `bottleneck <http://pypi.python.org/pypi/bottleneck/>`_ 0.7.0 or later.
* `futures <https://pypi.python.org/pypi/futures/>`_ 2.1.5 or later.
* `h5py <http://www.h5py.org/>`_ 2.2.1 or later.
* `matplotlib <http://matplotlib.org/>`_ 1.3.0 or later.
* `msgpack-numpy <http://pypi.python.org/pypi/msgpack-numpy>`_ 0.3.1.1 or later.
* `networkx <https://networkx.github.io>`_ 1.8 or later
* `numexpr <https://github.com/pydata/numexpr>`_ 2.3 or later.
* `numpy <http://numpy.scipy.org>`_ 1.2.0 or later.
* `pandas <http://pandas.pydata.org>`_ 0.14.1 or later.
* `pycuda <http://mathema.tician.de/software/pycuda>`_ 2012.1 or
  later.
* `pyzmq <http://zeromq.github.io/pyzmq/>`_ 13.0 or later.
* `scipy <http://www.scipy.org>`_ 0.11.0 or later.
* `tables <http://www.pytables.org>`_ 2.4.0 or later.
* `twiggy <http://twiggy.readthedocs.org/>`_ 0.4.0 or later.

If you have all of the above requirements installed, you can install 
the downloaded code using::

  cd neurokernel/
  python setup.py install

or (if you want to tinker with the code without having to repeatedly reinstall
it)::

  cd neurokernel/
  python setup.py develop
