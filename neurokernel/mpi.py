#!/usr/bin/env python

"""
MPI support classes.
"""

import inspect
import os
import re
import subprocess
import sys

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

from .mpi_proc import getargnames, Process, ProcessManager
from .mixins import LoggerMixin
from .tools.logging import setup_logger, set_excepthook
from .tools.misc import memoized_property, catch_exception

from tqdm import tqdm

class Worker(Process):
    """
    MPI worker class.

    This class repeatedly executes a work method.

    Parameters
    ----------
    ctrl_tag : int
        MPI tag to identify control messages transmitted to worker nodes.
    manager: bool
        Managerless running mode flag. It False, run Module without a
        manager. (default: True).
    """

    def __init__(self, ctrl_tag=1, manager = True, *args, **kwargs):
        super(Worker, self).__init__(manager = manager, *args, **kwargs)

        # Tag used to distinguish control messages:
        self._ctrl_tag = ctrl_tag
        # Execution step counter:
        self.steps = 0
        self.error = False

    # Define properties to perform validation when the maximum number of
    # execution steps set:
    _max_steps = float('inf')
    @property
    def max_steps(self):
        """
        Maximum number of steps to execute.
        """
        return self._max_steps
    @max_steps.setter
    def max_steps(self, value):
        if value < 0:
            raise ValueError('invalid maximum number of steps')
        self.log_info('maximum number of steps changed: %s -> %s' % \
                      (self._max_steps, value))
        self._max_steps = value

    def do_work(self):
        """
        Work method.

        This method is repeatedly executed by the Worker instance after the
        instance receives a 'start' control message and until it receives a 'stop'
        control message. It should be overridden by child classes.
        """

        self.log_info('executing do_work')

    def progressbar_name(self):
        return 'worker'

    def pre_run(self):
        """
        Code to run before main loop.

        This method is invoked by the `run()` method before the main loop is
        started.
        """
        self.log_info('running code before body of worker %s' % self.rank)
        self.post_run_complete = False

    def post_run(self):
        """
        Code to run after main loop.

        This method is invoked by the `run()` method after the main loop is
        started.
        """
        self._finalize()

    def _finalize(self):
        if not self.post_run_complete:
            self.pbar.close() # it should've already been closed in `run` but just to make sure.
            self.log_info('running code after body of worker %s' % self.rank)

            if self.manager:
                # Send acknowledgment message:
                self.intercomm.isend(['done', self.rank], 0, self._ctrl_tag)
                self.log_info('done message sent to manager')
            self.post_run_complete = True

    def catch_exception_run(self, func, *args, **kwargs):
        # If the debug flag is set, don't catch exceptions so that
        # errors will lead to visible failures:

        if self.manager:
            error = catch_exception(func, self.log_info, self.debug, *args, **kwargs)
            if error is not None:
                if not self.error:
                    self.intercomm.isend(['error', (self.id, self.steps, error)],
                                         dest=0, tag=self._ctrl_tag)
                    self.log_info('error sent to manager')
                    self.error = True
        else:
            error = catch_exception(func, self.log_info, True, *args, **kwargs)

    def run(self, steps = 0):
        """
        Main body of worker process.
        """

        #self.pre_run()
        self.catch_exception_run(self.pre_run)
        self.pbar = tqdm(desc = self.progressbar_name(), position = self.rank)

        self.log_info('running body of worker %s' % self.rank)

        # Start listening for control messages from parent process:
        if self.manager:
            r_ctrl = []
            try:
                d = self.intercomm.irecv(source=0, tag=self._ctrl_tag)
            except TypeError:
                # irecv() in mpi4py 1.3.1 stable uses 'dest' instead of 'source':
                d = self.intercomm.irecv(dest=0, tag=self._ctrl_tag)
            r_ctrl.append(d)
            req = MPI.Request()

        running = False
        self.steps = 0
        if not self.manager:
            self.max_steps = steps
            self.pbar.total = self.max_steps
            running = True
        while True:
            if self.manager:
                # Handle control messages (this assumes that only one control
                # message will arrive at a time):
                flag, msg_list = req.testall(r_ctrl)
                if flag:
                    msg = msg_list[0]

                    # Start executing work method:
                    if msg[0] == 'start':
                        self.log_info('starting')
                        running = True

                    # Stop executing work method::
                    elif msg[0] == 'stop':
                        if self.max_steps == float('inf'):
                            self.log_info('stopping')
                            running = False
                        else:
                            self.log_info('max steps set - not stopping')

                    # Set maximum number of execution steps:
                    elif msg[0] == 'steps':
                        if msg[1] == 'inf':
                            self.max_steps = float('inf')
                        else:
                            self.max_steps = int(msg[1])
                        self.pbar.total = self.max_steps
                        self.log_info('setting maximum steps to %s' % self.max_steps)

                    # Quit:
                    elif msg[0] == 'quit':
                        # if self.max_steps == float('inf'):
                        self.log_info('quitting')
                        break
                        # else:
                        #     self.log_info('max steps set - not quitting')

                    # Get next message:
                    r_ctrl = []
                    try:
                        d = self.intercomm.irecv(source=0, tag=self._ctrl_tag)
                    except TypeError:
                        # irecv() in mpi4py 1.3.1 stable uses 'dest' instead of 'source':
                        d = self.intercomm.irecv(dest=0, tag=self._ctrl_tag)
                    r_ctrl.append(d)

            # Execute work method; the work method may send data back to the master
            # as a serialized control message containing two elements, e.g.,
            # self.intercomm.isend(['foo', str(self.rank)],
            #                      dest=0, tag=self._ctrl_tag)
            if running:
                self.do_work()
                self.steps += 1
                self.pbar.update()
                self.log_info('execution step: %s' % self.steps)

            # Leave loop if maximum number of steps has been reached:
            if self.steps >= self.max_steps:
                running = False
                self.log_info('maximum steps reached')
                break

        #self.post_run()
        self.catch_exception_run(self.post_run)
        if not self.post_run_complete:
            self._finalize()

class WorkerManager(ProcessManager):
    """
    Self-launching MPI worker manager.

    This class may be used to construct an MPI application consisting of

    - a manager process that spawns MPI processes that execute the run() methods
      of several subclasses of the Worker class;
    - worker processes that perform some processing task; and

    The application should NOT be started via mpiexec.

    Parameters
    ----------
    ctrl_tag : int
        MPI tag to identify control messages transmitted to worker nodes.
        May not be equal to mpi4py.MPI.ANY_TAG

    Notes
    -----
    This class requires MPI-2 dynamic processing management.

    See Also
    --------
    Worker
    """

    def __init__(self, ctrl_tag=1):
        super(WorkerManager, self).__init__()

        # Validate control tag.
        assert ctrl_tag != MPI.ANY_TAG

        # Tag used to distinguish MPI control messages:
        self._ctrl_tag = ctrl_tag

    def add(self, target, *args, **kwargs):
        """
        Add a worker to an MPI application.

        Parameters
        ----------
        target : Worker
            Worker class to instantiate and run.
        args : sequence
            Sequential arguments to pass to target class constructor.
        kwargs : dict
            Named arguments to pass to target class constructor.
        """

        assert issubclass(target, Worker)
        self.log_info('adding class %s' % target.__name__)
        return ProcessManager.add(self, target, *args, **kwargs)

    def process_worker_msg(self, msg):
        """
        Process the specified deserialized message from a worker.
        """

        self.log_info('got ctrl msg: %s' % str(msg))

    def wait(self):
        """
        Wait for execution to complete.
        """

        # Start listening for control messages:
        r_ctrl = []
        try:
            d = self.intercomm.irecv(source=MPI.ANY_SOURCE,
                                     tag=self._ctrl_tag)
        except TypeError:
            # irecv() in mpi4py 1.3.1 stable uses 'dest' instead of 'source':
            d = self.intercomm.irecv(dest=MPI.ANY_SOURCE,
                                     tag=self._ctrl_tag)
        r_ctrl.append(d)
        workers = list(range(len(self)))
        req = MPI.Request()
        while True:
            # Check for control messages from workers:
            flag, msg_list = req.testall(r_ctrl)
            if flag:
                msg = msg_list[0]
                if msg[0] == 'done':
                    self.log_info('removing %s from worker list' % msg[1])
                    workers.remove(msg[1])

                # Additional control messages from the workers are processed
                # here:
                else:
                    self.process_worker_msg(msg)

                # Get new control messages:
                r_ctrl = []
                try:
                    d = self.intercomm.irecv(source=MPI.ANY_SOURCE,
                                             tag=self._ctrl_tag)
                except TypeError:
                    # irecv() in mpi4py 1.3.1 stable uses 'dest' instead of 'source':
                    d = self.intercomm.irecv(dest=MPI.ANY_SOURCE,
                                             tag=self._ctrl_tag)
                r_ctrl.append(d)

            if not workers:
                self.log_info('finished running manager')
                break

    def start(self, steps=float('inf')):
        """
        Tell the workers to start processing data.
        """

        self.log_info('sending steps message (%s)' % steps)
        for dest in range(len(self)):
            self.intercomm.isend(['steps', str(steps)], dest, self._ctrl_tag)
        self.log_info('sending start message')
        for dest in range(len(self)):
            self.intercomm.isend(['start'], dest, self._ctrl_tag)

    def stop(self):
        """
        Tell the workers to stop processing data.
        """

        self.log_info('sending stop message')
        for dest in range(len(self)):
            self.intercomm.isend(['stop'], dest, self._ctrl_tag)

    def quit(self):
        """
        Tell the workers to quit.
        """

        self.log_info('sending quit message')
        for dest in range(len(self)):
            self.intercomm.isend(['quit'], dest, self._ctrl_tag)

if __name__ == '__main__':
    import neurokernel.mpi_relaunch
    import time

    setup_logger(screen=True, file_name='neurokernel.log',
            mpi_comm=MPI.COMM_WORLD, multiline=True)

    # Define a class whose constructor takes arguments so as to test
    # instantiation of the class by the manager:
    class MyWorker(Worker):
        def __init__(self, x, y, z=None, routing_table=None):
            super(MyWorker, self).__init__()
            name = MPI.Get_processor_name()
            self.log_info('I am process %d of %d on %s.' % (self.rank,
                                                            self.size, name))
            self.log_info('init args: %s, %s, %s' % (x, y, z))

    man = WorkerManager()
    man.add(target=MyWorker, x=1, y=2, z=3)
    man.add(MyWorker, 3, 4, 5)
    man.add(MyWorker, 6, 7, 8)
    man.spawn()

    # To run for a specific number of steps, run
    # man.start(number_of_steps)
    man.start(100)
    man.wait()
