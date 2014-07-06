from pycuda.compiler import SourceModule
import pycuda.gpuarray as garray
import numpy as np

size_of_curandStateXORWOW = 12

def curand_setup(num_threads, seed):
    """
    Setup curand seed
    """
    func = get_curand_int_func()
    grid = ( (int(num_threads)-1)/128 + 1,1)
    block = (128,1,1)
    
    # curandStateXORWOW is a struct of size 12 4bytes
    state = garray.empty(num_threads*size_of_curandStateXORWOW, np.int32)
    
    func.prepared_call( grid, block, state.gpudata, num_threads, seed)
    return state


def get_curand_int_func():
    code = """
#include "curand_kernel.h"
extern "C" {
__global__ void 
rand_setup(curandStateXORWOW_t* state, int size, unsigned long long seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;

    for(int i = tid; i < size; i+=total_threads)
    {
        curand_init(seed, i, 0, &state[i]);
    }
}
}
    """
    mod = SourceModule(code, no_extern_c = True)
    func = mod.get_function("rand_setup")
    func.prepare([np.intp, np.int32, np.uint64])
    return func
    
    
