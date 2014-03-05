.. -*- rst -*-

Neurokernel
===========

Package Description
-------------------
Neurokernel is a Python framework for developing models of 
the fruit fly brain and executing them on multiple NVIDIA GPUs.

Quick Start
-----------
Make sure you have `pip <http://www.pip-installer.org>`_
installed; once you do, install the following dependencies 
as follows::

  pip install numpy
  pip install cython
  pip install numexpr
  pip install tables
  
Run the following to install the remaining dependencies and the 
latest Neurokernel code

  git clone https://github.com/neurokernel/neurokernel.git
  pip install -e git+./neurokernel#egg=neurokernel

If you have all of the requirements listed in ``INSTALL.rst`` installed, 
you can also install the downloaded code using the usual::

  python setup.py install

Check out the demos in ``neurokernel/examples`` subdirectory and 
their corresponding `IPython notebooks <http://ipython.org/notebook.html>`_ 
in ``neurokernel/notebooks``.

More Information
----------------
More information about Neurokernel can be obtained from
the project website at `<https://neurokernel.github.io>`_.

Authors & Acknowledgements
--------------------------
See the included AUTHORS file for more information.

License
-------
This software is licensed under the `BSD License
<http://www.opensource.org/licenses/bsd-license.php>`_.
See the included LICENSE file for more information.
