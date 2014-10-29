#!/usr/bin/env python

"""
MPI support classes.
"""

import inspect
import os
import re
import subprocess
import sys

from mpi4py import MPI
import msgpack
import psutil
import shortuuid
import twiggy
import zmq

from tools.logging import format_name, setup_logger, set_excepthook

def getargnames(f):
    """
    Get names of a callable's arguments.

    Parameters
    ----------
    f : callable
        Function to examine.

    Results
    -------
    args : list of str
        Argument names.

    Notes
    -----
    For instance methods, the `self` argument is omitted.
    """

    spec = inspect.getargspec(f)
    if inspect.ismethod(f):
        return spec.args[1:]
    else:
        return spec.args

def args_to_dict(f, *args, **kwargs):
    """
    Combine sequential and named arguments in single dictionary.

    Parameters
    ----------
    f : callable
        Function to which the arguments will be passed.
    args : tuple
        Sequential arguments.
    kwargs : dict
        Named arguments.

    Returns
    -------
    d : dict
        Maps argument names to values.
    """

    spec = inspect.getargspec(f)
    d = {}

    arg_names = getargnames(f)
    assert len(arg_names) <= args
    for arg, val in zip(arg_names, args):
        d[arg] = val
    for arg, val in kwargs.iteritems():
        if arg in d:
            raise ValueError('\'%s\' already specified in positional args' % arg)
        d[arg] = val
    return d

class PollerChecker(object):
    """
    Wrapper class to facilitate creation and use of ZMQ pollers.

    Parameters
    ----------
    sock : zmq.Socket
        ZeroMQ socket to poll.
    direction : int
        Polling direction.
    """

    def __init__(self, sock, direction=zmq.POLLIN):
        self._sock = sock
        self._poller = zmq.Poller()
        self._poller.register(sock, direction)

    def check(self, timeout=None):
        """
        Check for I/O.

        Parameters
        ----------
        timeout : float, int
            Timeout in milliseconds. If None, no timeout is assumed.

        Returns
        -------
        status : bool
            True if transmitted messages are available.
        """

        socks = dict(self._poller.poll(timeout))
        if self._sock in socks:
            return True
        else:
            return False

class Worker(object):
    """
    MPI worker class.

    This class repeatedly executes a work method.

    Parameters
    ----------
    data_tag : int
        MPI tag to identify data messages transmitted between worker nodes.
    ctrl_tag : int
        MPI tag to identify control messages transmitted to worker nodes.
    """

    def __init__(self, data_tag=0, ctrl_tag=1):
        rank = MPI.COMM_WORLD.Get_rank()
        self.logger = twiggy.log.name(format_name('worker %s' % rank))
        set_excepthook(self.logger, True)

        # Tags used to distinguish MPI messages:
        self._data_tag = data_tag
        self._ctrl_tag = ctrl_tag

        # Maximum number of execution steps:
        self.steps = float('inf')

    def do_work(self):
        """
        Work method.

        This method is repeatedly executed by the Worker instance after the
        instance receives a 'start' control message and until it receives a 'stop'
        control message. It should be overridden by child classes.
        """

        self.logger.info('executing do_work')

    def run(self):
        """
        Main body of worker process.
        """

        rank = MPI.COMM_WORLD.Get_rank()
        self.logger.info('running body of worker %s' % rank)

        # Start listening for control messages:
        r_ctrl = []
        r_ctrl.append(MPI.COMM_WORLD.irecv(source=0, tag=self._ctrl_tag))

        running = False
        req = MPI.Request()
        steps = 0
        while True:

            # Handle control messages:
            flag, msg = req.testall(r_ctrl)
            if flag:

                # Deserialize:
                msg = msgpack.loads(msg[0])

                # Start listening for data messages:
                if msg[0] == 'start':
                    self.logger.info('started')
                    running = True

                # Stop listening for data messages:
                elif msg[0] == 'stop':
                    self.logger.info('stopped')
                    running = False

                # Set maximum number of execution steps:
                elif msg[0] == 'steps':
                    if msg[1] == 'inf':
                        self.steps = float('inf')
                    else:
                        self.steps = int(msg[1])
                    self.logger.info('steps set to %s' % self.steps)
                    
                # Quit:
                elif msg[0] == 'quit':
                    self.logger.info('acknowledging quit')

                    # Block until acknowledgment message received:
                    MPI.COMM_WORLD.send(str(rank), dest=0, tag=self._ctrl_tag)
                    self.logger.info('quit')
                    return

                r_ctrl = []
                r_ctrl.append(MPI.COMM_WORLD.irecv(source=0, tag=self._ctrl_tag))

            # Execute work method:
            if running:
                self.do_work()
                steps += 1
                self.logger.info('execution step: %s' % steps)

            # Send acknowledgment back to master if a finite number of steps was
            # specified and then completed:
            if steps > self.steps:
                self.logger.info('acknowledging step')

                # Block until acknowledgment message received:
                MPI.COMM_WORLD.send(str(rank), dest=0, tag=self._ctrl_tag)
                self.logger.info('step')
                return

class Manager(object):
    """
    Self-launching MPI worker manager.

    This class may be used to construct an MPI application consisting of

    - a launcher process with methods for specifying Worker class instances as
      the bodies of MPI nodes and for starting and stopping the application's
      execution;
    - worker processes that perform some processing task; and
    - a master process that relays control messages from the launcher process to the
      worker processes.

    The application should NOT be started via mpiexec.

    Parameters
    ----------
    mpiexec : str
        Name of MPI launcher executable.
    mpiargs : tuple
        Additional arguments to pass to MPI launcher.
    data_tag : int
        MPI tag to identify data messages transmitted between worker nodes.
    ctrl_tag : int
        MPI tag to identify control messages transmitted to worker nodes.

    Notes
    -----
    This class does not require MPI-2 dynamic processing management.

    See Also
    --------
    Worker
    """

    def __init__(self, mpiexec='mpiexec', mpiargs=(), data_tag=0, ctrl_tag=1):
        assert data_tag != ctrl_tag and data_tag != MPI.ANY_TAG and \
                           ctrl_tag != MPI.ANY_TAG

        # MPI launch info:
        self._mpiexec = mpiexec
        self._mpiargs = tuple(mpiargs)

        # Make logger name reflect process identity:
        if self._is_launcher():
            self.logger = twiggy.log.name(format_name('launcher/manager'))
        elif self._is_master():
            self.logger = twiggy.log.name(format_name('master/manager'))
        else:
            self.logger = twiggy.log.name(format_name('worker %s/manager' % MPI.COMM_WORLD.Get_rank()))
        set_excepthook(self.logger, True)

        # Tags used to distinguish MPI messages:
        self._data_tag = data_tag
        self._ctrl_tag = ctrl_tag

        # Worker classes to instantiate:
        self._targets = {}

        # Arguments to pass to worker class constructors:
        self._kwargs = {}

        # Reserve node 0 for use as master:
        self._rank = 1

    @property
    def num_workers(self):
        """
        Number of workers known to manager.
        """

        return len(self._targets)

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

        Returns
        -------
        rank : int
            MPI rank assigned to class.
        """

        # The contents of this method need to run on the launcher (so that it
        # knows how many MPI processes to start), the workers (so that they
        # know how to instantiate their respective classes), and the master (so
        # that it will be aware of all of the MPI processes):
        self.logger.info('adding class %s' % target.__name__)
        assert issubclass(target, Worker)

        rank = self._rank
        self._targets[rank] = target
        self._kwargs[rank] = args_to_dict(target.__init__, *args, **kwargs)
        self._rank += 1
        return rank

    def _in_mpi(self):
        """
        Return True if the current process is an MPI process.
        """

        # Get name of parent process:
        parent_name = psutil.Process(os.getppid()).name()

        # All MPI processes must be children of the launcher process:
        return bool(re.search(os.path.basename(self._mpiexec), parent_name))

    def _is_launcher(self):
        """
        Return True if the current process is the launching process.
        """

        return not self._in_mpi()

    def _is_master(self):
        """
        Return True if the current process is the master MPI process.
        """

        # We need to check if MPI is active because Get_rank() will return zero 
        # in the launcher too:
        return self._in_mpi() and (MPI.COMM_WORLD.Get_rank() == 0)

    def _run_launcher(self):
        """
        Asynchronously launch MPI application, connect master and launcher with ZeroMQ.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping launch')
            return
        else:
            self.logger.info('launching application (%s)' % self._rank)

        # Create random IPC interface name:
        env = os.environ.copy()
        env['MASTER_IPC_INT'] = 'ipc://'+shortuuid.uuid()

        # Set up communication socket to master process:
        ctx = zmq.Context()
        self._sock = ctx.socket(zmq.ROUTER)
        self._sock.bind(env['MASTER_IPC_INT'])
        self._pc = PollerChecker(self._sock, zmq.POLLIN)

        # Pass the IPC interface name to the launched process via an
        # environmental variable:
        python_path = sys.executable
        script_name = os.path.basename(sys.argv[0])
        self._mpiexec_proc = subprocess.Popen((self._mpiexec,)+self._mpiargs+\
                                              ('-np', str(self._rank),
                                               python_path, script_name),
                                              stdout=sys.stdout,
                                              stderr=sys.stderr,
                                              stdin=sys.stdin,
                                              env=env)
        self.logger.info('application launched')

        # Synchronize connection:
        while True:
            self._sock.send_multipart(['master', ''])
            if self._pc.check(10):
                self._sock.recv_multipart()
                break
        self.logger.info('launcher synchronized')

    def _run_master(self):
        """
        Body of master MPI process.

        The only function of the master process is to relay messages from the
        launcher to the other nodes.
        """

        if not self._is_master():
            self.logger.info('not in master - skipping _run_master')
            return
        else:
            self.logger.info('running body of master')

        ctx = zmq.Context()
        self._sock = ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.IDENTITY, 'master')
        self._sock.connect(os.environ['MASTER_IPC_INT'])
        self._pc = PollerChecker(self._sock, zmq.POLLIN)

        # Synchronize connection:
        while True:
            if self._pc.check(10):
                self._sock.recv()
                self._sock.send('')
                break
        self.logger.info('master synchronized')

        # Relay messages from launcher to workers until a quit or step message is received:
        # XXX currently only broadcasts messages to workers; could be extended
        # to permit directed transmission to specific workers:
        size = MPI.COMM_WORLD.Get_size()
        while True:

            # Check for messages from launcher:
            if self._pc.check(10):
                msg = self._sock.recv_multipart()

                # Pass any messages on to all of the workers:
                self.logger.info('sending message to workers: '+str(msg))
                for i in xrange(1, size):
                    MPI.COMM_WORLD.isend(msgpack.dumps(msg),
                                         dest=i, tag=self._ctrl_tag)

                # Wait for all workers to acknowledge quit or step message:
                if msg[0] == 'quit' or msg[0] == 'step':
                    req = MPI.Request()
                    r_quit = []
                    self.logger.info('waiting for quit/step acknowledgments')
                    for i in xrange(1, size):
                        r_quit.append(MPI.COMM_WORLD.irecv(source=MPI.ANY_SOURCE,
                                                           tag=self._ctrl_tag))
                    while not req.testall(r_quit):
                        pass

                    self.logger.info('finished running master')
                    break

    def run(self):
        """
        Run MPI application.

        Notes
        -----
        This method must be called after all callable code that is to be run on
        the MPI nodes has been added to the manager. This method will return
        immediately on the launcher but not on the master or worker processes.
        """

        if self._is_launcher():
            self._run_launcher()
        elif self._is_master():
            self._run_master()
        else:

            # Instantiate each target using the specified parameters and
            # execute the target's run() method:
            rank = MPI.COMM_WORLD.Get_rank()
            t = self._targets[rank](**self._kwargs[rank])
            t.run()
            self.logger.info('finished running %s' % rank)

    def start(self):
        """
        Tell the workers to start processing data.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping start')
            return
        self.logger.info('sending start message')
        self._sock.send_multipart(['master', 'start'])

    def stop(self):
        """
        Tell the workers to stop processing data.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping stop')
            return
        self.logger.info('sending stop message')
        self._sock.send_multipart(['master', 'stop'])

    def steps(self, n=float('inf')):
        """
        Tell the workers to run no more than the specified number of steps.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping steps')
            return
        self.logger.info('sending steps message (%s)' % n)
        self._sock.send_multipart(['master', 'steps', str(n)])

    def quit(self):
        """
        Tell the workers to quit.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping quit')
            return
        self.logger.info('sending quit message')
        self._sock.send_multipart(['master', 'quit'])

    def kill(self):
        """
        Kill MPI launcher.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping kill')
            return
        self.logger.info('killing launcher')
        self._mpiexec_proc.kill()

    def wait(self):
        """
        Wait for MPI launcher to exit.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping wait')
            return
        self.logger.info('waiting for launcher to exit')
        self._mpiexec_proc.wait()

    def terminate(self):
        """
        Terminate MPI launcher.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping')
            return
        self.logger.info('terminating launcher')
        self._mpiexec_proc.terminate()

if __name__ == '__main__':
    import time

    setup_logger(stdout=sys.stdout)

    # Define a class whose constructor takes arguments so as to test
    # instantiation of the class by the manager:
    class MyWorker(Worker):
        def __init__(self, x, y, z=None):
            super(MyWorker, self).__init__()
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            name = MPI.Get_processor_name()
            self.logger.info('I am process %d of %d on %s.' % (rank, size, name))
            self.logger.info('init args: %s, %s, %s' % (x, y, z))

    man = Manager()
    man.add(target=MyWorker, x=1, y=2, z=3)
    man.add(MyWorker, 3, 4, 5)
    man.run()
    man.start()
    time.sleep(1)
    man.stop()
    man.quit()
