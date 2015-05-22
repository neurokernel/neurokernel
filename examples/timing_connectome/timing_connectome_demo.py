#!/usr/bin/env python

"""
Create and run multiple empty LPUs to time data reception throughput.

Notes
-----
* Requires connectivity matrix in Document S2 from 
  http://dx.doi.org/10.1016/j.cub.2015.03.021
* Requires CUDA 7.0 when using MPS because of multi-GPU support.
* The maximum allowed number of open file descriptors must be sufficiently high.
"""

import argparse
import glob
import itertools
import numbers
import os
import re
import sys
import time
import warnings

try:
    import cudamps
except ImportError:
    mps_avail = False
else:
    mps_avail = True
from mpi4py import MPI
import networkx as nx
import numpy as np
import pandas as pd
import pycuda.driver as drv
import pymetis
import twiggy

from neurokernel.all_global_vars import all_global_vars
from neurokernel.core_gpu import CTRL_TAG, GPOT_TAG, SPIKE_TAG, Manager, Module
from neurokernel.pattern import Pattern
from neurokernel.plsel import Selector, SelectorMethods
from neurokernel.tools.logging import setup_logger

class MyModule(Module):
    """
    Empty module class.

    This module class doesn't do anything in its execution step apart from
    transmit/receive dummy data. All spike ports are assumed to
    produce/consume data at every step.
    """

    def __init__(self, sel, sel_in, sel_out,
                 sel_gpot, sel_spike, data_gpot, data_spike,
                 columns=['interface', 'io', 'type'],
                 ctrl_tag=CTRL_TAG, gpot_tag=GPOT_TAG, spike_tag=SPIKE_TAG,
                 id=None, device=None,
                 routing_table=None, rank_to_id=None,
                 debug=False, time_sync=False):
        if data_gpot is None:
            data_gpot = np.zeros(SelectorMethods.count_ports(sel_gpot), float)
        if data_spike is None:
            data_spike = np.zeros(SelectorMethods.count_ports(sel_spike), int)
        super(MyModule, self).__init__(sel, sel_in, sel_out,
                 sel_gpot, sel_spike, data_gpot, data_spike,
                 columns,
                 ctrl_tag, gpot_tag, spike_tag,
                 id, device,
                 routing_table, rank_to_id,
                 debug, time_sync)

        self.pm['gpot'][self.interface.out_ports().gpot_ports(tuples=True)] = 1.0
        self.pm['spike'][self.interface.out_ports().spike_ports(tuples=True)] = 1

class MyManager(Manager):
    """
    Manager that can use Multi-Process Service.

    Parameters
    ----------
    use_mps : bool
        If True, use Multi-Process Service so that multiple MPI processes
        can use the same GPUs concurrently.
    """

    def __init__(self, use_mps=False):
        super(MyManager, self).__init__()

        if use_mps:
            self._mps_man = cudamps.MultiProcessServiceManager()
        else:
            self._mps_man = None

    def spawn(self, part_map):
        """
        Spawn MPI processes for and execute each of the managed targets.

        Parameters
        ----------
        part_map : dict
            Maps GPU ID to list of target MPI ranks.
        """

        if self._is_parent:

            # The number of GPUs over which the targets are partitioned may not
            # exceed the actual number of supported devices:
            n_part_gpus = len(part_map.keys())
            n_avail_gpus = 0
            drv.init()
            for i in xrange(drv.Device.count()):

                # MPS requires Tesla/Quadro GPUs with compute capability 3.5 or greater:
                if mps_avail:
                    d = drv.Device(i)
                    if d.compute_capability() >= (3, 5) and \
                       re.search('Tesla|Quadro', d.name()):
                        n_avail_gpus += 1
                else:
                    n_avail_gpus += 1
            if n_part_gpus > n_avail_gpus:
                raise RuntimeError('partition size (%s) exceeds '
                                   'number of available GPUs (%s)' % \
                                   (n_part_gpus, n_avail_gpus))
                
            # Start MPS control daemons (this assumes that the available GPUs
            # are numbered consecutively from 0 onwards - as are the elements of
            # part_map.keys()):
            if self._mps_man:
                self._mps_man.start()
                self.log_info('starting MPS')

            # Find the path to the mpi_backend.py script (which should be in the
            # same directory as this module:
            import neurokernel.mpi
            parent_dir = os.path.dirname(neurokernel.mpi.__file__)
            mpi_backend_path = os.path.join(parent_dir, 'mpi_backend.py')

            # Check that the union ranks in the partition correspond exactly to 
            # those of the targets added to the manager:
            n_targets = len(self._targets.keys())
            if set(self._targets.keys()) != \
               set([t for t in itertools.chain.from_iterable(part_map.values())]):
                raise ValueError('partition must contain all target ranks')

            # Invert mapping of GPUs to MPI ranks:
            rank_to_gpu_map = {rank:gpu for gpu in part_map.keys() for rank in part_map[gpu]}

            # Set MPS pipe directory:
            info = MPI.Info.Create()
            if self._mps_man:
                mps_dir = self._mps_man.get_mps_dir(self._mps_man.get_mps_ctrl_proc())
                info.Set('env', 'CUDA_MPS_PIPE_DIRECTORY=%s' % mps_dir)

            # Spawn processes:
            self._intercomm = MPI.COMM_SELF.Spawn(sys.executable,
                                                  args=[mpi_backend_path],
                                                  maxprocs=n_targets,
                                                  info=info)

            # First, transmit twiggy logging emitters to spawned processes so
            # that they can configure their logging facilities:
            for i in self._targets.keys():
                self._intercomm.send(twiggy.emitters, i)

            # Transmit class to instantiate, globals required by the class, and
            # the constructor arguments; the backend will wait to receive
            # them and then start running the targets on the appropriate nodes.
            for i in self._targets.keys():
                target_globals = all_global_vars(self._targets[i])

                # Serializing atexit with dill appears to fail in virtualenvs
                # sometimes if atexit._exithandlers contains an unserializable function:
                if 'atexit' in target_globals:
                    del target_globals['atexit']
                data = (self._targets[i], target_globals, self._kwargs[i])
                self._intercomm.send(data, i)

    def __del__(self):
        # Shut down MPS daemon when the manager is cleaned up:
        if self._mps_man:
            pid = self._mps_man.get_mps_ctrl_proc()
            self.log_info('stopping MPS control daemon %i' % pid)
            self._mps_man.stop(pid)
        
def gen_sels(conn_mat, scaling=1):
    """
    Generate port selectors for LPUs in benchmark test.

    Parameters
    ----------
    conn_mat : numpy.ndarray
        Square array containing numbers of directed spiking port connections 
        between LPUs (which correspond to the row and column indices). 
    scaling : int
        Scaling factor; multiply all connection numbers by this value.

    Returns
    -------
    mod_sels : dict of tuples
        Ports in module interfaces; the keys are the module IDs and the values are tuples
        containing the respective selectors for all ports, all input ports, all
        output ports, all graded potential, and all spiking ports.
    pat_sels : dict of tuples
        Ports in pattern interfaces; the keys are tuples containing the two
        module IDs connected by the pattern and the values are pairs of tuples
        containing the respective selectors for all source ports, all
        destination ports, all input ports connected to the first module, 
        all output ports connected to the first module, all graded potential ports 
        connected to the first module, all spiking ports connected to the first
        module, all input ports connected to the second module, 
        all output ports connected to the second  module, all graded potential ports 
        connected to the second module, and all spiking ports connected to the second
        module.
    """

    conn_mat = np.asarray(conn_mat)
    r, c = conn_mat.shape
    assert r == c
    n_lpu = r

    assert scaling > 0 and isinstance(scaling, numbers.Integral)
    conn_mat *= scaling

    # Construct selectors describing the ports exposed by each module:
    mod_sels = {}
    for i in xrange(n_lpu):
        lpu_id = 'lpu%s' % i

        # Structure ports as 
        # /lpu_id/in_or_out/spike_or_gpot/other_lpu_id/[0:n_spike]
        # where in_or_out is relative to module i:
        sel_in_gpot = Selector('')
        sel_out_gpot = Selector('')

        sel_in_spike = \
            Selector(','.join(['/lpu%i/in/spike/lpu%i/[0:%i]' % (i, j, n) for j, n in \
                               enumerate(conn_mat[:, i]) if (j != i and n != 0)]))
        sel_out_spike = \
            Selector(','.join(['/lpu%i/out/spike/lpu%i/[0:%i]' % (i, j, n) for j, n in \
                               enumerate(conn_mat[i, :]) if (j != i and n != 0)]))
                                                                         
        mod_sels[lpu_id] = (Selector.union(sel_in_gpot, sel_in_spike,
                                           sel_out_gpot, sel_out_spike),
                            Selector.union(sel_in_gpot, sel_in_spike),
                            Selector.union(sel_out_gpot, sel_out_spike),
                            Selector.union(sel_in_gpot, sel_out_gpot),
                            Selector.union(sel_in_spike, sel_out_spike))

    # Construct selectors describing the ports connected by each pattern:
    pat_sels = {}
    for i, j in itertools.combinations(xrange(n_lpu), 2):
        lpu_i = 'lpu%s' % i
        lpu_j = 'lpu%s' % j

        # The pattern's input ports are labeled "../out.." because that selector
        # describes the output ports of the connected module's interface:
        sel_in_gpot_i = Selector('')
        sel_out_gpot_i = Selector('')
        sel_in_gpot_j = Selector('')
        sel_out_gpot_j = Selector('')

        sel_in_spike_i = Selector('/%s/out/spike/%s[0:%i]' % (lpu_i, lpu_j,
                                                              conn_mat[i, j]))
        sel_out_spike_i = Selector('/%s/in/spike/%s[0:%i]' % (lpu_i, lpu_j,
                                                              conn_mat[j, i]))
        sel_in_spike_j = Selector('/%s/out/spike/%s[0:%i]' % (lpu_j, lpu_i,
                                                              conn_mat[j, i]))
        sel_out_spike_j = Selector('/%s/in/spike/%s[0:%i]' % (lpu_j, lpu_i,
                                                              conn_mat[i, j]))

        # The order of these two selectors is important; the individual 'from'
        # and 'to' ports must line up properly for Pattern.from_concat to
        # produce the right pattern:
        sel_from = Selector.add(sel_in_gpot_i, sel_in_spike_i,
                                sel_in_gpot_j, sel_in_spike_j)
        sel_to = Selector.add(sel_out_gpot_j, sel_out_spike_j,
                              sel_out_gpot_i, sel_out_spike_i)

        # Exclude scenarios where the "from" or "to" selector is empty (and
        # therefore cannot be used to construct a pattern):
        if len(sel_from) and len(sel_to):
            pat_sels[(lpu_i, lpu_j)] = \
                    (sel_from, sel_to,
                     Selector.union(sel_in_gpot_i, sel_in_spike_i),
                     Selector.union(sel_out_gpot_i, sel_out_spike_i),
                     Selector.union(sel_in_gpot_i, sel_out_gpot_i),
                     Selector.union(sel_in_spike_i, sel_out_spike_i),
                     Selector.union(sel_in_gpot_j, sel_in_spike_j),
                     Selector.union(sel_out_gpot_j, sel_out_spike_j),
                     Selector.union(sel_in_gpot_j, sel_out_gpot_j),
                     Selector.union(sel_in_spike_j, sel_out_spike_j))

    return mod_sels, pat_sels

def partition(mat, n_parts):
    """
    Partition a directed graph described by a weighted connectivity matrix.
    
    Parameters
    ----------
    mat : numpy.ndarray
        Square weighted connectivity matrix for a directed graph.
    n_parts : int
        Number of partitions.

    Returns
    -------
    part_map : dict of list
        Dictionary of partitions. The dict keys are the partition identifiers,
        and the values are the lists of nodes in each partition.
    """
    
    # Combine weights of directed edges to obtain undirected graph:
    mat = mat+mat.T

    # Convert matrix into METIS-compatible form:
    g = nx.from_numpy_matrix(np.array(mat, dtype=[('weight', int)])) 
    n = g.number_of_nodes()
    e = g.number_of_edges()
    xadj = np.empty(n+1, int)
    adjncy = np.empty(2*e, int)
    eweights = np.empty(2*e, int)
    end_node = 0
    xadj[0] = 0
    for i in g.node:
        for j, a in g.edge[i].items():
            adjncy[end_node] = j
            eweights[end_node] = a['weight']
            end_node += 1
        xadj[i+1] = end_node

    # Compute edge-cut partition:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cutcount, part_vert = pymetis.part_graph(n_parts, xadj=xadj,
                                                 adjncy=adjncy, eweights=eweights)

    # Find nodes in each partition:
    part_map = {}
    for i, p in enumerate(set(part_vert)):
        ind = np.where(np.array(part_vert) == p)[0]
        part_map[p] = ind
    return part_map

def emulate(conn_mat, scaling, n_gpus, steps, use_mps):
    """
    Benchmark inter-LPU communication throughput.

    Each LPU is configured to use a different local GPU.

    Parameters
    ----------
    conn_mat : numpy.ndarray
        Square array containing numbers of directed spiking port connections 
        between LPUs (which correspond to the row and column indices). 
    scaling : int
        Scaling factor; multiply all connection numbers by this value.
    n_gpus : int
        Number of GPUs over which to partition the emulation.
    steps : int
        Number of steps to execute.
    use_mps : bool
        Use Multi-Process Service if True.

    Returns
    -------
    average_throughput, total_throughput : float
        Average per-step and total received data throughput in bytes/seconds.
    exec_time : float
        Execution time in seconds.
    """

    # Time everything starting with manager initialization:
    start_all = time.time()

    # Set up manager:
    man = MyManager(use_mps)

    # Generate selectors for configuring modules and patterns:
    mod_sels, pat_sels = gen_sels(conn_mat, scaling)

    # Partition nodes in connectivity matrix:
    part_map = partition(conn_mat, n_gpus)

    # Set up modules such that those in each partition use that partition's GPU:
    ranks = set([rank for rank in itertools.chain.from_iterable(part_map.values())])
    rank_to_gpu_map = {rank:gpu for gpu in part_map for rank in part_map[gpu]}
    for i in ranks:
        lpu_i = 'lpu%s' % i
        sel, sel_in, sel_out, sel_gpot, sel_spike = mod_sels[lpu_i]
        man.add(MyModule, lpu_i, sel, sel_in, sel_out, sel_gpot, sel_spike,
                None, None, ['interface', 'io', 'type'],
                CTRL_TAG, GPOT_TAG, SPIKE_TAG, device=rank_to_gpu_map[i],
                time_sync=True)

    # Set up connections between module pairs:
    for lpu_i, lpu_j in pat_sels.keys():
        sel_from, sel_to, sel_in_i, sel_out_i, sel_gpot_i, sel_spike_i, \
            sel_in_j, sel_out_j, sel_gpot_j, sel_spike_j = pat_sels[(lpu_i, lpu_j)]
        pat = Pattern.from_concat(sel_from, sel_to,
                                  from_sel=sel_from, to_sel=sel_to, data=1, validate=False)
        pat.interface[sel_in_i, 'interface', 'io'] = [0, 'in']
        pat.interface[sel_out_i, 'interface', 'io'] = [0, 'out']
        pat.interface[sel_gpot_i, 'interface', 'type'] = [0, 'gpot']
        pat.interface[sel_spike_i, 'interface', 'type'] = [0, 'spike']
        pat.interface[sel_in_j, 'interface', 'io'] = [1, 'in']
        pat.interface[sel_out_j, 'interface', 'io'] = [1, 'out']
        pat.interface[sel_gpot_j, 'interface', 'type'] = [1, 'gpot']
        pat.interface[sel_spike_j, 'interface', 'type'] = [1, 'spike']
        man.connect(lpu_i, lpu_j, pat, 0, 1, compat_check=False)

    man.spawn(part_map)
    start_main = time.time()
    man.start(steps)
    man.wait()
    stop_main = time.time()
    return man.average_step_sync_time, (time.time()-start_all), (stop_main-start_main), \
        (man.stop_time-man.start_time)

if __name__ == '__main__':
    import neurokernel.mpi_relaunch

    conn_mat_file = 's2.xlsx'
    scaling = 1
    max_steps = 100
    n_gpus = 4
    use_mps = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False,
                        dest='debug', action='store_true',
                        help='Enable debug mode.')
    parser.add_argument('-l', '--log', default='none', type=str,
                        help='Log output to screen [file, screen, both, or none; default:none]')
    parser.add_argument('-c', '--conn_mat_file', default=conn_mat_file, type=str,
                        help='Connectivity matrix Excel file [default: %s]' % conn_mat_file)
    parser.add_argument('-k', '--scaling', default=scaling, type=int,
                        help='Connection number scaling factor [default: %s]' % scaling)
    parser.add_argument('-m', '--max_steps', default=max_steps, type=int,
                        help='Maximum number of steps [default: %s]' % max_steps)
    parser.add_argument('-g', '--gpus', default=n_gpus, type=int,
                        help='Number of GPUs [default: %s]' % n_gpus)
    parser.add_argument('-p', '--use_mps', action='store_true',
                        help='Use Multi-Process Service [default: False]')
    args = parser.parse_args()

    file_name = None
    screen = False
    if args.log.lower() in ['file', 'both']:
        file_name = 'neurokernel.log'
    if args.log.lower() in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen,
                          mpi_comm=MPI.COMM_WORLD,
                          multiline=True)

    conn_mat = pd.read_excel('s2.xlsx',
                             sheetname='Connectivity Matrix').astype(int).as_matrix()
    
    #conn_mat = conn_mat[0:25, 0:25]
    #N = 10
    #conn_mat = 200*(np.ones((N, N), dtype=int)-np.eye(N, dtype=int))
    print emulate(conn_mat, args.scaling, args.gpus, args.max_steps, args.use_mps)
