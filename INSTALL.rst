.. -*- rst -*-

Installation
============

Prerequisites
-------------
Neurokernel requires Python 2.7, at least one NVIDIA GPU, NVIDIA's `GPU drivers 
<http://www.nvidia.com/content/drivers/>`_, and `CUDA 
<http://www.nvidia.com/object/cuda_home_new.html>`_ 5.0 or later.  To check what 
GPUs are in your system, you can use the `inxi 
<https://code.google.com/p/inxi/>`_ command available on most Linux 
distributions::

  inxi -G

You can verify that the drivers are loaded as follows::

  lsmod | grep nvidia

If no drivers are present, you may have to manually load them by running 
something like::

  modprobe nvidia

as root.

Quick Start
-----------
Make sure you have `pip <http://pip.pypa.io>`_ installed (preferably
in a `virtualenv <http://virtualenv.pypa.io>`_); once you do, install the
following dependencies as follows::

  pip install numpy
  pip install cython
  pip install numexpr
  pip install tables
  pip install pycuda

If installation of PyCUDA fails because some of the CUDA development files or 
libraries are not found, you may need to specify where they are explicitly. For 
example, if CUDA is installed in ``/usr/local/cuda/``, try installing PyCUDA
as follows::

  CUDA_ROOT=/usr/local/cuda/ CFLAGS=-I${CUDA_ROOT}/include \
  LDFLAGS=-L${CUDA_ROOT}/lib64 pip install pycuda

Replace ``${CUDA_ROOT}/lib`` with ``${CUDA_ROOT}/lib64`` if your system is 
running 64-bit
Linux. If you continue to encounter installation problems, see the `PyCUDA Wiki 
<http://wiki.tiker.net/PyCuda/Installation>`_ for more information.

You will also need to have `ffmpeg <http://www.fmpeg.org>`_ or `libav 
<http://libav.org>`_ installed to generate some of the demo visualizations.
  
Run the following to install the remaining dependencies and the 
latest Neurokernel code::

  git clone https://github.com/neurokernel/neurokernel.git
  pip install -e git+./neurokernel#egg=neurokernel

Supported Platforms
-------------------
Neurokernel has been tested and installed on Linux. It may run on other
platforms too; if you encounter problems, submit a bug report on
`GitHub <https://github.com/neurokernel/neurokernel/issues>`_.

Installation Dependencies
-------------------------
Neurokernel currently requires the following Python packages:

* `bidict <http://pypi.python.org/pypi/bidict/>`_ 0.1.0 or later.
* `bottleneck <http://pypi.python.org/pypi/bottleneck/>`_ 0.7.0 or later.
* `futures <https://pypi.python.org/pypi/futures/>`_ 2.1.5 or later.
* `h5py <http://www.h5py.org/>`_ 2.2.1 or later.
* `lxml <http://lxml.de/>`_ 3.3.0 or later.
* `matplotlib <http://matplotlib.org/>`_ 1.3.0 or later.
* `msgpack-numpy <http://pypi.python.org/pypi/msgpack-numpy>`_ 0.3.1.1 or later.
* `networkx <https://networkx.github.io>`_ 1.9 or later.
* `numexpr <https://github.com/pydata/numexpr>`_ 2.3 or later.
* `numpy <http://numpy.scipy.org>`_ 1.2.0 or later.
* `pandas <http://pandas.pydata.org>`_ 0.14.1 or later.
* `ply <http://www.dabeaz.com/ply/>`_ 3.4 or later.
* `pycuda <http://mathema.tician.de/software/pycuda>`_ 2014.1 or later.
* `pyzmq <http://zeromq.github.io/pyzmq/>`_ 13.0 or later.
* `scipy <http://www.scipy.org>`_ 0.11.0 or later.
* `tables <http://www.pytables.org>`_ 2.4.0 or later.
* `twiggy <http://twiggy.readthedocs.org/>`_ 0.4.0 or later.

Building the HTML documentation locally requires

* `sphinx_rtd_theme <https://github.com/snide/sphinx_rtd_theme>`_ 0.1.6 or 
  later.

If you have all of the above requirements installed, you can install 
the downloaded code using::

  cd neurokernel/
  python setup.py install

or (if you want to tinker with the code without having to repeatedly reinstall
it)::

  cd neurokernel/
  python setup.py develop
