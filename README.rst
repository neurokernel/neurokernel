.. -*- rst -*-

..  image:: https://raw.githubusercontent.com/neurokernel/neurokernel/master/docs/source/_static/logo.png
    :alt: Neurokernel

Package Description
-------------------

`Project Website <https://neurokernel.github.io>`_ |
`GitHub Repository <https://github.com/neurokernel/neurokernel>`_ |
`Online Documentation <https://neurokernel.readthedocs.io>`_ |
`Mailing List <https://lists.columbia.edu/mailman/listinfo/neurokernel-dev>`_ |
`Forum <http://neurokernel.67426.x6.nabble.com/>`_

Neurokernel is a Python framework for developing models of
the fruit fly brain and executing them on multiple NVIDIA GPUs.

.. image:: http://prime4commit.com/projects/98.svg
    :target: http://prime4commit.com/projects/98
    :alt: Support the project

Prerequisites
-------------
Neurokernel requires

* Linux (other operating systems may work, but have not been tested);
* Python;
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
install OpenMPI. See `this page
<https://www.open-mpi.org/faq/?category=building#easy-build>`_
for OpenMPI installation information. *Note that OpenMPI 1.8* |openmpi_no_windows|_.

.. _openmpi_no_windows: https://www.open-mpi.org/software/ompi/v1.6/ms-windows.php
.. |openmpi_no_windows| replace:: *cannot run on Windows*

Some of Neurokernel's demos require either `ffmpeg <http://www.fmpeg.org>`_ or `libav
<http://libav.org>`_ installed to generate visualizations (see `Examples`_).

Installation
------------

Conda
^^^^^
The easiest way to get neurokernel is to install it in a conda environment: ::

  conda create -n nk python=3.7 c-compiler compilers cxx-compiler openmpi -c conda-forge -y
  conda activate nk
  python -m pip install neurokernel

Make sure to enable CUDA support in the installed OpenMPI by setting: ::

  export OMPI_MCA_opal_cuda_support=true

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

Authors & Acknowledgements
--------------------------
See the included `AUTHORS`_ file for more information.

.. _AUTHORS: AUTHORS.rst

License
-------
This software is licensed under the `BSD License
<http://www.opensource.org/licenses/bsd-license.php>`_.
See the included `LICENSE`_ file for more information.

.. _LICENSE: LICENSE.rst

Notes
-----
The Neurokernel Project is independent of the NeuroKernel Operating System
developed by `NeuroDNA Computer <http://www.neurokernel.com>`_.
