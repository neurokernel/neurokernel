.. -*- rst -*-

Introduction
============

Neurokernel is an open software platform written in Python for emulation of the
brain of the fruit fly (*Drosophila melanogaster*) on multiple Graphics
Processing Units (GPUs). It provides a programming model based upon the
organization of the fly's brain into fewer than 50 modular subdivisions called
*local processing units* (LPUs) that are each characterized by unique
populations of local neurons [1]_. Using Neurokernel's API, researchers can develop
models of individual LPUs and combine them with other independently developed
LPU models to collaboratively construct models of entire subsystems of the fly
brain. Neurokernel's support for LPU model integration also enables exploration
of brain functions that cannot be exclusively attributed to individual LPUs or
brain subsystems.

Examples of Neurokernel's use are available on the `project website
<http://neurokernel.github.io/docs>`_.

.. [1] Chiang, A.-S., Lin, C.-Y., Chuang, C.-C., Chang, H.-M., Hsieh, C.-H., Yeh,
       C.-W., et al. (2011), Three-dimensional reconstruction of brain-wide wiring
       networks in Drosophila at single-cell resolution, Current Biology, 21, 1, 1–11,
       `doi:10.1016/j.cub.2010.11.056 <http://www.cell.com/current-biology/abstract/S0960-9822%2810%2901522-8>`_
