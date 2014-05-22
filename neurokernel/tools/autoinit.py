#!/usr/bin/env python

"""
Autoinitialize multiple GPUs.
"""

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.tools as tools
import atexit

class MultiGPUManager(object):
    """
    Create and manage contexts for multiple GPUs.

    Parameters
    ----------
    gpus : list of int
        IDs of GPUs for which to create contexts. If no IDs are
        specified, create contexts for all GPUs on the system.

    Methods
    -------
    switch(gpu)
        Make the context associated with the specified GPU active.

    Notes
    -----
    After instantiation, the context associated with the last
    specified GPU is active.

    """

    def __init__(self, *gpus):
        N = drv.Device(0).count()
        if len(gpus) == 0:
            gpus = range(N)
        if max(gpus) > N-1:
            raise ValueError('nonexistent GPU specified')

        self._curr_gpu = None
        self.dev_dict = {}
        self.ctx_dict = {}
        for gpu in gpus:
            dev = drv.Device(gpu)
            self.dev_dict[gpu] = dev
            ctx = dev.make_context()
            self.ctx_dict[gpu] = ctx
            def cleanup():
                ctx.pop()
                tools.clear_context_caches()
            atexit.register(cleanup)
        self._curr_gpu = gpu

    @property
    def curr_gpu(self):
        """
        Return GPU associated with currently active context.
        """

        return self._curr_gpu

    def switch_gpu(self, gpu):
        """
        Switch to the context associated with the specified GPU.
        """

        if not self.ctx_dict.has_key(gpu):
            raise ValueError('nonexistent GPU specified')

        if gpu != self.curr_gpu:
            self.ctx_dict[self._curr_gpu].pop()
            self.ctx_dict[gpu].push()
            self._curr_gpu = gpu

drv.init()
global gpu_ctx_manager
gpu_ctx_manager = MultiGPUManager()
curr_gpu = gpu_ctx_manager.curr_gpu
switch_gpu = gpu_ctx_manager.switch_gpu

if __name__ == '__main__':
    import numpy as np
    man = MultiGPUManager()
    x_gpu = gpuarray.to_gpu(np.array([1, 2, 3]))
    man.switch_gpu(0)
    y_gpu = gpuarray.to_gpu(np.array([4, 5, 6]))
    man.switch_gpu(1)
    print x_gpu
    man.switch_gpu(0)
    print y_gpu

    # This will cause an error:
    print x_gpu
