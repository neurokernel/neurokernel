.. -*- rst -*

Neurokernel Design 
==================

A Neurokernel brain emulation comprises two main conceptual units:

* *Module objects* contain models of the various Local Processing Units (LPUs)
  or hubs comprised by the *Drosophila* brain. These models are characterized by
  the presence of local neurons that do not project to neurons in other modules
  and projection neurons that do emit output to neurons in other modules.
* *Connectivity objects* describe how neurons in two different module are connected
  to each other. Apart from representing the presence and direction of
  connections between neurons, they contain the parameters associated with each
  connection.

Each module's interface must expose the neurons that emit output for destination
neurons or receive input from source neurons in other modules. Neurons may be
assumed to emit either graded potentials at each step of an emulation or spikes.
Exposed graded-potential and spiking neurons are identified by separate
sequences of integers. Neurokernel currently does not provide any built-in
neuron or synapse implementations; the user of the software is responsible for
implementing the models comprised by each module.

Connectivity objects must be compatible with the modules they connect - that is,
the number of graded potential and spiking neurons exposed by each of the
connected modules must equivalent to the corresponding number of neurons assumed
by the connectivity object's two interfaces.

Once the connectivity between modules is defined, Neurokernel automatically
transmits updated neuron membrane potentials and generated spikes between
modules. Instantiation and execution of the entire emulation is also managed by
Neurokernel.


