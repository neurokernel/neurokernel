#!/usr/bin/env python

import time
from unittest import main, TestCase

import zmq

from neurokernel.tools.comm import get_random_port
from neurokernel.ctrl_proc import ControlledProcess

class test_ctrl_proc(TestCase):
    def setUp(self):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.ROUTER)
        self.port = self.sock.bind_to_random_port('tcp://*')
        self.proc = ControlledProcess(self.port, 'proc')

    def tearDown(self):
        self.proc.terminate()
        
    def test_ctrl_quit(self):    
        self.proc.start()
        time.sleep(1)
        self.sock.send_multipart([self.proc.id, 'quit'])
        time.sleep(1)
        assert not self.proc.is_alive()

if __name__ == '__main__':
    main()
