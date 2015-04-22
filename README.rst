.. -*- rst -*-

..  image:: https://raw.githubusercontent.com/neurokernel/neurokernel/master/docs/source/_static/logo.png
    :alt: Neurokernel

Package Description
-------------------
Neurokernel is a Python framework for developing models of 
the fruit fly brain and executing them on multiple NVIDIA GPUs.

.. image:: http://prime4commit.com/projects/98.svg
    :target: http://prime4commit.com/projects/98
    :alt: Support the project

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
following dependencies in the specified order (replace `pip` with the path to 
the copy of `pip` in your `virtualenv`)::

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

You need to have `OpenMPI <http://www.open-mpi.org>`_ 1.8.4 or later compiled 
with CUDA support installed. See `this page 
<https://www.open-mpi.org/faq/?category=building#easy-build>`_ for installation 
information.

You will also need to have `ffmpeg <http://www.fmpeg.org>`_ or `libav 
<http://libav.org>`_ installed to generate some of the demo visualizations.

Run the following to install the remaining Python package dependencies listed in 
`setup.py` and the latest Neurokernel code: ::

  git clone https://github.com/neurokernel/neurokernel.git
  pip install -e git+./neurokernel#egg=neurokernel

You can also install the code directly as follows: ::

  cd neurokernel/
  python setup.py install

or (if you want to tinker with the code without having to repeatedly reinstall
it)::

  cd neurokernel/
  python setup.py develop

Building the Documentation
--------------------------
To build the HTML documentation, you will need to install 

* `sphinx <http://sphinx-doc.org>`_ 1.2 or later.
* `sphinx_rtd_theme <https://github.com/snide/sphinx_rtd_theme>`_ 0.1.6 or 
  later.
 
Once these are installed, run the following: ::

  cd neurokernel/docs
  make html

Supported Platforms
-------------------
Neurokernel has been tested and installed on Linux. It may run on other
platforms too; if you encounter problems, submit a bug report on
`GitHub <https://github.com/neurokernel/neurokernel/issues>`_.

More Information
----------------
More information about Neurokernel can be obtained from
the project website at `<https://neurokernel.github.io>`_.

Neurokernel's documentation is available at `<http://neurokernel.rtfd.org>`_.

Authors & Acknowledgements
--------------------------
See the included AUTHORS file for more information.

License
-------
This software is licensed under the `BSD License
<http://www.opensource.org/licenses/bsd-license.php>`_.
See the included LICENSE file for more information.
