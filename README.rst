.. -*- rst -*-

Neurokernel
===========

Package Description
-------------------
Neurokernel is a Python framework for developing models of 
the fruit fly brain and executing them on multiple NVIDIA GPUs.

.. image:: http://prime4commit.com/projects/98.svg
    :target: http://prime4commit.com/projects/98
    :alt: Support the project

Quick Start
-----------
Neurokernel requires Python 2.7, at least one NVIDIA GPU, NVIDIA's `GPU drivers 
<http://www.nvidia.com/content/drivers/>`_, and `CUDA 
<http://www.nvidia.com/object/cuda_home_new.html>`_ 5.0 or later.

Make sure you have `pip <http://pip.pypa.io>`_ installed (preferably
in a `virtualenv <http://virtualenv.pypa.io>`_); once you do, install the
following dependencies as follows::

  pip install numpy
  pip install cython
  pip install numexpr
  pip install tables
  pip install pycuda

If installation of `pycuda` fails because some of the CUDA development files or 
libraries are not found, you may need to specify where they are explicitly. For 
example, if CUDA is installed in `/usr/local/cuda/`, try installing `pycuda` as 
follows::

  CUDA_ROOT=/usr/local/cuda/ CFLAGS=-I${CUDA_ROOT}/include \
  LDFLAGS=-L${CUDA_ROOT}/lib64 pip install pycuda

Replace `${CUDA_ROOT}/lib` with `${CUDA_ROOT}/lib64` if your system is running 64-bit
Linux. If you continue to encounter installation problems, see the `PyCUDA Wiki 
<http://wiki.tiker.net/PyCuda/Installation>`_ for more information.

You will also need to have `ffmpeg <http://www.fmpeg.org>`_ or `libav 
<http://libav.org>`_ installed to generate some of the demo visualizations.

Run the following to install the remaining dependencies and the latest 
Neurokernel code::

  git clone https://github.com/neurokernel/neurokernel.git
  pip install -e git+./neurokernel#egg=neurokernel

If you have all of the requirements listed in ``INSTALL.rst`` installed, 
you can also install the downloaded code using::

  cd neurokernel/
  python setup.py install

or (if you want to tinker with the code without having to repeatedly reinstall
it)::

  cd neurokernel/
  python setup.py develop

Check out the demos in ``neurokernel/examples`` subdirectory and 
their corresponding `IPython notebooks <http://ipython.org/notebook.html>`_ 
in ``neurokernel/notebooks``.

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
