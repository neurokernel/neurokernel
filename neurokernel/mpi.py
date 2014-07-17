#!/usr/bin/env python

"""
MPI support classes.
"""

import os
import re
import subprocess
import sys

from mpi4py import MPI
import psutil
import shortuuid
import twiggy
import zmq
import zmq.log.handlers

from tools.comm import ZMQOutput

def format_name(name, width=20):
    """
    Pad process name with spaces.

    Parameters
    ----------
    name : str
        Name to pad.
    width : int
        Total width of padded name.

    Returns
    -------
    padded : str
        Padded name.
    """

    return ('{name:%s}' % width).format(name=name)

def setup_logger(name='', level=twiggy.levels.DEBUG,
                 fmt=twiggy.formats.line_format,
                 stdout=None, file_name=None, sock=None):
    """
    Setup a twiggy logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : twiggy.levels.LogLevel
        Logging level.
    fmt : str
        Format string.
    stdout : bool
        Create output stream handler to stdout if True.
    file_name : str
        Create output handler to specified file.
    sock : str
        ZeroMQ socket address.

    Returns
    -------
    logger : twiggy.logger.Logger
        Configured logger.
    """

    if file_name:
        file_output = \
          twiggy.outputs.FileOutput(file_name, fmt, 'w')
        twiggy.addEmitters(('file', level, None, file_output))

    if stdout:
        stdout_output = \
          twiggy.outputs.StreamOutput(fmt, stream=stdout)   
        twiggy.addEmitters(('stdout', level, None, stdout_output))

    if sock:
        port_output = ZMQOutput(sock, fmt)
        twiggy.addEmitters(('sock', level, None, sock_output))

    return twiggy.log.name(format_name(name))

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

        # Tags used to distinguish MPI messages:
        self._data_tag = data_tag
        self._ctrl_tag = ctrl_tag

    def run_step(self):
        """
        Work method.

        This method is repeatedly executed by the Worker instance after the
        instance receives a 'start' control message and until it receives a 'stop'
        control message. It should check for the presence of 
        """

        self.logger.info('executing run_step')

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
        while True:

            # Handle control messages:
            flag, msg = req.testall(r_ctrl)
            if flag:

                # Start listening for data messages:
                if msg[0] == 'start':
                    self.logger.info('started')
                    running = True

                # Stop listening for data messages:
                elif msg[0] == 'stop':
                    self.logger.info('stopped')
                    running = False

                # Quit:
                elif msg[0] == 'quit':
                    self.logger.info('quitting')
                    return

                r_ctrl = []
                r_ctrl.append(MPI.COMM_WORLD.irecv(source=0, tag=self._ctrl_tag))

            # Execute work method:
            if running:
                self.run_step()

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

    This class does not require MPI-2 dynamic processing management.

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
            self.logger = twiggy.log.name(format_name('manager/launcher  '))
        elif self._is_master():
            self.logger = twiggy.log.name(format_name('manager/master    '))
        else:
            self.logger = twiggy.log.name(format_name('manager/worker %s ' % MPI.COMM_WORLD.Get_rank()))

        # Tags used to distinguish MPI messages:
        self._data_tag = data_tag
        self._ctrl_tag = ctrl_tag
        self._targets = {}
        self._args = {}
        self._kwargs = {}

        # Reserve node 0 for use as master:
        self._rank = 1

    def add(self, target=None, args=(), kwargs={}):
        """
        Add a worker to an MPI application.

        Parameters
        ----------
        target : Worker
            Worker class to instantiate and run.
        args : list
            Sequential arguments to pass to target class constructor.
        kwargs : dict
            Named arguments to pass to target class constructor.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping add')
        else:
            self.logger.info('adding worker')

        assert issubclass(target, Worker)
        self._targets[self._rank] = target
        self._args[self._rank] = tuple(args)
        self._kwargs[self._rank] = kwargs
        self._rank += 1

    def _is_launcher(self):
        """
        Return True if the current process is the launching process.
        """

        # Get name of parent process:
        parent_name = psutil.Process(os.getppid()).name()

        # All processes launched by the master process must be children of the
        # launcher:
        return not bool(re.search(os.path.basename(self._mpiexec), parent_name))

    def _is_master(self):
        """
        Return True if the current process is the master MPI process.
        """
        
        return MPI.COMM_WORLD.Get_rank() == 0

    def _run_launcher(self):
        """
        Asynchronously launch MPI application, connect master and launcher with ZeroMQ.
        """

        if not self._is_launcher():
            self.logger.info('not in launcher - skipping launch')
            return
        else:
            self.logger.info('launching application')

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
        script_name = os.path.basename(__file__)
        self._mpiexec_proc = subprocess.Popen((self._mpiexec,)+self._mpiargs+\
                                              ('-np', str(self._rank), python_path, script_name),
                                              stdout=sys.stdout,
                                              stderr=sys.stderr,
                                              stdin=sys.stdin,
                                              env=env)

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

        # Relay messages from launcher to workers until a quit message is received:
        # XXX currently only broadcasts messages to workers; could be extended
        # to permit directed transmission to specific workers:
        size = MPI.COMM_WORLD.Get_size()
        while True:

            # Check for messages from launcher:
            if self._pc.check(10):
                msg = self._sock.recv()

                # Pass any messages on to all of the workers:
                for i in xrange(1, size):
                    MPI.COMM_WORLD.isend(msg, dest=i, tag=self._ctrl_tag)

                if msg == 'quit':
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
            t = self._targets[rank](*self._args[rank], **self._kwargs[rank])
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
            raise RuntimeError('not in launcher - skipping')
        self.logger.info('terminating launcher')
        self._mpiexec_proc.terminate()

if __name__ == '__main__':
    import time

    setup_logger(stdout=sys.stdout)

    class MyWorker(Worker):
        def __init__(self):
            super(MyWorker, self).__init__()
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            name = MPI.Get_processor_name()
            self.logger.info('I am process %d of %d on %s.' % (rank, size, name))
            
    man = Manager()
    man.add(target=MyWorker)
    man.add(target=MyWorker)
    man.run()
    man.start()
    time.sleep(1)
    man.stop()
    man.quit()
