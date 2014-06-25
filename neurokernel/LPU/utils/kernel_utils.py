#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.gpuarray as garray
import numpy as np


def launch_kernel(func, block, grid, argins, timed = None, shared = None, texrefs=[], prepared = False):
    
    
    Nargin = len(argins)
    arg_type = []
    arg_call = []
    
    for arg in range(Nargin):
        a = argins[arg]
        if a.__class__ == tuple or a.__class__ == list:
            b = a[0]
            offset = a[1]
            #print b.__class__
            if b.__class__ == garray.GPUArray:
                arg_type.append(np.intp)
                
                arg_call.append(np.intp(int(b.gpudata) + offset * b.dtype.itemsize))
            
            elif b.__class__.__name__ == 'PitchArray':
                arg_type.append(np.intp)
                
                arg_call.append(np.intp(int(b.gpudata) + offset * b.dtype.itemsize))
            elif b.__class__ == cuda.DeviceAllocation:
                arg_type.append(np.intp)
                arg_call.append(np.intp(int(b) + offset))
                
            elif b.__class__ in [int, np.int32, np.int64, np.int16]:
                if offset == 'u':
                    arg_type.append(np.uint32)
                    arg_call.append(np.uint32(b))
                else:
                    arg_type.append(np.int32)
                    arg_call.append(np.int32(b))
                    
            elif b.__class__ in [np.uint64, np.uint32, np.uint16]:
                if offset == 'u':
                    arg_type.append(np.uint32)
                    arg_call.append(np.uint32(b))
                else:
                    arg_type.append(np.int32)
                    arg_call.append(np.int32(b))
            
            elif b.__class__ in [float, np.float32, np.float64]:
                if offset == 'f':
                    arg_type.append(np.float32)
                    arg_call.append(np.float32(b))
                else:
                    arg_type.append(np.float64)
                    arg_call.append(np.float64(b))
            
            elif b.__class__ in [complex, np.complex64, np.complex128]:
                if offset == 'f':
                    arg_type.append(np.complex64)
                    arg_call.append(np.complex64(b))
                else:
                    arg_type.append(np.complex128)
                    arg_call.append(np.complex128(b))
            else:
                print "warning: unknown type"
                arg_type.append(a.__class__)
                arg_call.append(a)
                
            
        elif a.__class__ == garray.GPUArray:
            arg_type.append(np.intp)
            arg_call.append(a.gpudata)
        elif a.__class__.__name__ == 'PitchArray':
            arg_type.append(np.intp)
            arg_call.append(a.gpudata)
            
        elif a.__class__ == cuda.DeviceAllocation:
            arg_type.append(np.intp)
            arg_call.append(a)
            
        elif a.__class__ in [int, np.int32, np.int64, np.int16]:
            arg_type.append(np.int32)
            arg_call.append(np.int32(a))
            
        elif a.__class__ in [float, np.float64]:
            arg_type.append(np.float64)
            arg_call.append(np.float64(a))
            
        elif a.__class__ == np.float32:
            arg_type.append(np.float32)
            arg_call.append(a)
            
        elif a.__class__ in [np.uint64, np.uint32, np.uint16]:
            arg_type.append(np.uint32)
            arg_call.append(np.uint32(a))
            
        elif a.__class__ in [complex, np.complex128]:
            arg_type.append(np.complex128)
            arg_call.append(np.complex128(a))
            
        elif a.__class__ == np.complex64:
            arg_type.append(np.complex64)
            arg_call.append(a)
            
        else:
            print "launch kernel warning: unknown type"
            arg_type.append(a.__class__)
            arg_call.append(a)

    if not prepared:
        func.prepare(arg_type, texrefs = texrefs)
    
    new_grid = (int(grid[0]), int(grid[1]))
    new_block = (int(block[0]), int(block[1]), int(block[2]))
    
    if timed is None:
        func.prepared_call(new_grid, new_block, *arg_call, shared_size = 0 if shared is None else shared)
        
    else:
        a = func.prepared_timed_call(new_grid, new_block, *arg_call, shared_size = 0 if shared is None else shared)
        if timed == 1:
            return a()
        else:
            time = a()
            print "%s returned in: %f ms" % (timed, time * 1000)
            


def func_compile(func_name, source, options=["--ptxas-options=-v"], keep=False, no_extern_c=False, arch=None, code=None, cache_dir=None):
    from pycuda.compiler import SourceModule
    
    mod = SourceModule(source, options = options, keep = keep, no_extern_c= no_extern_c, arch=arch, code=code, cache_dir=cache_dir)
    func = mod.get_function(func_name)
    
    return func



