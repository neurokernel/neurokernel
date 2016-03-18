.. -*- rst -*-

Installation
============

Prerequisites
-------------
Neurokernel requires

* Linux (other operating systems may work, but have not been tested);
* Python 2.7 (Python 3.0 is not guaranteed to work);
* at least one NVIDIA GPU with `Fermi
  <http://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf>`_
  architecture or later;
* NVIDIA's `GPU drivers <http://www.nvidia.com/content/drivers/>`_;
* `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`_ 5.0 or later;
* `OpenMPI <http://www.open-mpi.org>`_ 1.8.4 or later compiled with CUDA support.

To check what GPUs are in your system, you can use the `inxi
<https://code.google.com/p/inxi/>`_ command available on most Linux
distributions::

  inxi -G

You can verify that the drivers are loaded as follows::

  lsmod | grep nvidia

If no drivers are present, you may have to manually load them by running
something like::

  modprobe nvidia

as root.

Although some Linux distributions do include CUDA in their stock package
repositories, you are encouraged to use those distributed by NVIDIA because they
often are more up-to-date and include more recent releases of the GPU drivers.
See `this page <https://developer.nvidia.com/cuda-downloads>`_ for download
information.

If you install Neurokernel in a virtualenv environment, you will need to
install OpenMPI. See `this page <https://www.open-mpi.org/faq/?category=building#easy-build>`_
for OpenMPI installation information. *Note that OpenMPI 1.8* |openmpi_no_windows|_.

.. _openmpi_no_windows: https://www.open-mpi.org/software/ompi/v1.6/ms-windows.php
.. |openmpi_no_windows| replace:: *cannot run on Windows*

Some of Neurokernel's demos require either `ffmpeg <http://www.fmpeg.org>`_ or `libav
<http://libav.org>`_ installed to generate visualizations (see `Examples`_).

Installation
------------
Download the latest Neurokernel code as follows: ::

  git clone https://github.com/neurokernel/neurokernel.git

Since Neurokernel requires a fair number of additional Python packages to run,
it is recommended that it either be installed in a `virtualenv
<http://www.virtualenv.org/>`_ or `conda <http://conda.io/>`_
environment. Follow the relevant instructions below.

Virtualenv
^^^^^^^^^^
See `this page <https://virtualenv.pypa.io/en/latest/installation.html>`_ for
virtualenv installation information.

Create a new virtualenv environment and install several required dependencies: ::

  cd ~/
  virtualenv NK
  ~/NK/bin/pip install numpy cython numexpr pycuda

If installation of PyCUDA fails because some of the CUDA development files or
libraries are not found, you may need to specify where they are explicitly. For
example, if CUDA is installed in ``/usr/local/cuda/``, try installing PyCUDA
as follows::

  CUDA_ROOT=/usr/local/cuda/ CFLAGS=-I${CUDA_ROOT}/include \
  LDFLAGS=-L${CUDA_ROOT}/lib64 ~/NK/bin/pip install pycuda

Replace ``${CUDA_ROOT}/lib`` with ``${CUDA_ROOT}/lib64`` if your system is
running 64-bit Linux. If you continue to encounter installation problems, see
the `PyCUDA Wiki <http://wiki.tiker.net/PyCuda/Installation>`_ for more information.

Run the following to install the remaining Python package dependencies listed in
`setup.py`: ::

  cd ~/neurokernel
  ~/NK/bin/python setup.py develop

Conda
^^^^^
*Note that conda packages are currently only available for 64-bit Ubuntu Linux
14.04. If you would like packages for another distribution, please submit a
request to the* |nk_developers|_.

.. _nk_developers: http://github.com/neurokernel/neurokernel/issues
.. |nk_developers| replace:: *Neurokernel developers*

First, install the following Ubuntu packages:

* ``libibverbs1``
* ``libnuma1``
* ``libpmi0``
* ``libslurm26``
* ``libtorque2``
  
Tthese are required by the conda OpenMPI packages prepared
for Neurokernel. Ensure that the stock Ubuntu OpenMPI packages are not installed
because they may interfere with the ones that will be installed by conda. You 
also need to ensure that CUDA has been installed in
``/usr/local/cuda``.

Install conda by either installing `Anaconda
<https://store.continuum.io/cshop/anaconda/>`_
or `Miniconda <http://conda.pydata.org/miniconda.html>`_. Make sure that the
following lines appear in your `~/.condarc` file so that conda can find the
packages required by Neurokernel: ::

   channels:
   - https://conda.binstar.org/neurokernel/channel/ubuntu1404
   - defaults

Create a new conda environment containing the packages required by Neurokernel
by running the following command: ::

   conda create -n NK neurokernel_deps

PyCUDA packages compiled against several versions of CUDA are available. If you
need one compiled against a specific version that differs from the one
automatically installed by the above command, you will need to manually install
it afterwards as follows (replace ``cuda75`` with the appropriate version): ::

  source activate NK
  conda install pycuda=2015.1.3=np110py27_cuda75_0
  source deactivate

Activate the new environment and install Neurokernel in it as follows: ::

  source activate NK
  cd ~/neurokernel
  python setup.py develop

Examples
--------
Introductory examples of how to use Neurokernel to build and integrate models of different
parts of the fly brain are available in the `Neurodriver
<https://github.com/neurokernel/neurodriver>`_ package. To install it run the
following: ::

  git clone https://github.com/neurokernel/neurodriver
  cd ~/neurodriver
  python setup.py develop

Other models built using Neurokernel are available on
`GitHub <https://github.com/neurokernel/>`_.

Building the Documentation
--------------------------
To build Neurokernel's HTML documentation locally, you will need to install

* `mock <http://www.voidspace.org.uk/python/mock/>`_ 1.0 or later.
* `sphinx <http://sphinx-doc.org>`_ 1.3 or later.
* `sphinx_rtd_theme <https://github.com/snide/sphinx_rtd_theme>`_ 0.1.6 or
  later.

Once these are installed, run the following: ::

  cd ~/neurokernel/docs
  make html
