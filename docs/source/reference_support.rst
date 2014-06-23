.. -*- rst -*-

Support Classes and Functions
=============================

Path-Like Port Identifier Handling
----------------------------------
.. currentmodule:: neurokernel.plsel
.. autosummary::
   :toctree: generated/
   :nosignatures:

   PathLikeSelector
   PortMapper

Context Managers
----------------
.. currentmodule:: neurokernel.ctx_managers
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ExceptionOnSignal
   IgnoreKeyboardInterrupt
   IgnoreSignal
   OnKeyboardInterrupt
   TryExceptionOnSignal

Communication Tools
-------------------
.. currentmodule:: neurokernel.tools.comm
.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_random_port
   is_poll_in
   ZMQOutput

.. reenable after these are rewritten to use the new Interface/Pattern classes
   Graph Tools
   -----------
   .. currentmodule:: neurokernel.tools.graph
   .. autosummary::
      :toctree: generated/
      :nosignatures:

      graph_to_conn
      graph_to_df
      load_conn_all

Visualization Tools
-------------------
.. currentmodule:: neurokernel.tools.plot
.. autosummary::
   :toctree: generated/
   :nosignatures:

   imdisp
   show_pydot
   show_pygraphviz

Other
-----
.. currentmodule:: neurokernel.tools.misc
.. autosummary::
   :toctree: generated/
   :nosignatures:

   catch_exception
   rand_bin_matrix
