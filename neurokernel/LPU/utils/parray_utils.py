#!/usr/bin/env python


import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from pycuda.tools import dtype_to_ctype
from kernel_utils import func_compile




def get_fill_function(dtype, pitch = True):
    type_dst = dtype_to_ctype(dtype)
    name = "fill"
    
    if pitch:
        func = func_compile(name, fill_pitch_template % {"name": name,
                       "type_dst": type_dst})
        func.prepare([np.int32, np.int32, np.intp, np.int32, dtype.type])
    else:
        func = func_compile(name, fill_nonpitch_template % {"name": name,
                       "type_dst": type_dst})
        func.prepare([np.int32, np.intp, dtype.type])
    return func


def get_astype_function(dtype_dest, dtype_src, pitch = True):
    type_dest = dtype_to_ctype(dtype_dest)
    type_src = dtype_to_ctype(dtype_src)
    name = "astype"
    
    operation = ""
    
    if pitch:
        func = func_compile(name, pitch_template % {"name": name,
                   "dest_type": type_dest,
                   "src_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32])
    else:
        func = func_compile(name, non_pitch_template % {"name": name,
                   "dest_type": type_dest,
                   "src_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, np.int32])
    return func


    
def get_realimag_function(dtype, real = True, pitch = True):
    type_src = dtype_to_ctype(dtype)
    
    if dtype == np.complex64:
        type_dest = "float"
        if real:
            operation = "pycuda::real"
            name = "real"
        else:
            operation = "pycuda::imag"
            name = "imag"
    elif dtype == np.complex128:
        type_dest = "double"
        if real:
            operation = "pycuda::real"
            name = "real"
        else:
            operation = "pycuda::imag"
            name = "imag"
    else:
        raise TypeError("only support complex inputs as numpy.complex64 or numpy.complex128")
    
    if pitch:
        func = func_compile(name, pitch_template % {"name": name,
                   "dest_type": type_dest,
                   "src_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32])
    else:
        func = func_compile(name, non_pitch_template % {"name": name,
                   "dest_type": type_dest,
                   "src_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, np.int32])
    return func


    



def get_abs_function(dtype, pitch = True):
    type_src = dtype_to_ctype(dtype)

    if dtype == np.complex128:
        operation = "pycuda::abs"
        type_dest = "double"
    elif dtype == np.complex64:
        operation = "pycuda::abs"
        type_dest = "float"
    elif dtype == np.float64:
        operation = "fabs"
        type_dest = "double"
    elif dtype == np.float32:
        operation = "fabsf"
        type_dest = "float"
    else:
        operation = "abs"
        type_dest = dtype_to_ctype(dtype)
    
    name = "abs_function"
    
    
    if pitch:
        func = func_compile(name, pitch_template % {"name": name,
                   "dest_type": type_dest,
                   "src_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32])
    else:
        func = func_compile(name, non_pitch_template % {"name": name,
                   "dest_type": type_dest,
                   "src_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, np.int32])
    
    return func


def get_conj_function(dtype, pitch = True):
    
    type_src = dtype_to_ctype(dtype)
    if dtype == np.complex128:
        operation = "pycuda::conj"
    elif dtype == np.complex64:
        operation = "pycuda::conj"
    else:
        raise TypeError("Only complex arrays are allowed to perform conjugation")
    
    name = "conj"
    
    if pitch:
        func = func_compile(name, pitch_template % {"name": name,
                   "dest_type": type_src,
                   "src_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32])
    else:
        func = func_compile(name, non_pitch_template % {"name": name,
                   "dest_type": type_src,
                   "src_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, np.int32])
    
    return func



def get_resize_function(dtype):
    type_src = dtype_to_ctype(dtype)
    name = "resize"
    
    func = func_compile(name, reshape_template % {"name": name,
                   "dest_type": type_src,
                   "src_type": type_src,
                   "operation": "",
                   })
    func.prepare([np.int32, np.int32, np.int32, np.int32, np.intp, np.int32, np.intp, np.int32])
    return func


def get_transpose_function(dtype, conj = False):

    src_type = dtype_to_ctype(dtype)
    name = "trans"
    operation = ""
    if conj:
        if dtype == np.complex128:
            operation = "pycuda::conj"
        elif dtype == np.complex64:
            operation = "pycuda::conj"
    
    func = func_compile(name, transpose_template % {"name": name, "type": src_type, "operation": operation})
    func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32])
    return func

def get_addarray_function(left_dtype, right_dtype, rslt_dtype, pitch = True):
    type_left = dtype_to_ctype(left_dtype)
    type_right = dtype_to_ctype(right_dtype)
    type_rslt = dtype_to_ctype(rslt_dtype)

    name = "addarray"
    operation = "+"

    if pitch:
        func = func_compile(name, pitch_array_op_template % {"name": name,
                   "dest_type": type_rslt,
                   "left_type": type_left,
                   "right_type": type_right,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, np.intp, np.int32])
    else:
        func = func_compile(name, non_pitch_array_op_template % {"name": name,
                   "dest_type": type_rslt,
                   "left_type": type_left,
                   "right_type": type_right,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, np.intp, np.int32])
    return func

def get_addscalar_function(src_type, pitch = True):
    type_src = dtype_to_ctype(src_type)

    name = "addscalar"
    operation = "+"
    
    if pitch:
        func = func_compile(name, pitch_left_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, src_type.type])
    else:
        func = func_compile(name, non_pitch_left_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, src_type.type, np.int32])
    return func

def get_subarray_function(left_dtype, right_dtype, rslt_dtype, pitch = True):
    type_left = dtype_to_ctype(left_dtype)
    type_right = dtype_to_ctype(right_dtype)
    type_rslt = dtype_to_ctype(rslt_dtype)

    name = "subdarray"
    operation = "-"

    if pitch:
        func = func_compile(name, pitch_array_op_template % {"name": name,
                   "dest_type": type_rslt,
                   "left_type": type_left,
                   "right_type": type_right,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, np.intp, np.int32])
    else:
        func = func_compile(name, non_pitch_array_op_template % {"name": name,
                   "dest_type": type_rslt,
                   "left_type": type_left,
                   "right_type": type_right,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, np.intp, np.int32])
    return func

def get_subscalar_function(src_type, pitch = True):
    type_src = dtype_to_ctype(src_type)

    name = "subscalar"
    operation = "-"
    
    if pitch:
        func = func_compile(name, pitch_left_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, src_type.type])
    else:
        func = func_compile(name, non_pitch_left_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, src_type.type, np.int32])
    return func

def get_scalarsub_function(src_type, pitch = True):
    type_src = dtype_to_ctype(src_type)

    name = "scalarsub"
    operation = "-"
    
    if pitch:
        func = func_compile(name, pitch_right_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, src_type.type])
    else:
        func = func_compile(name, non_pitch_right_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, src_type.type, np.int32])
    return func

def get_mularray_function(left_dtype, right_dtype, rslt_dtype, pitch = True):
    type_left = dtype_to_ctype(left_dtype)
    type_right = dtype_to_ctype(right_dtype)
    type_rslt = dtype_to_ctype(rslt_dtype)

    name = "mularray"
    operation = "*"

    if pitch:
        func = func_compile(name, pitch_array_op_template % {"name": name,
                   "dest_type": type_rslt,
                   "left_type": type_left,
                   "right_type": type_right,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, np.intp, np.int32])
    else:
        func = func_compile(name, non_pitch_array_op_template % {"name": name,
                   "dest_type": type_rslt,
                   "left_type": type_left,
                   "right_type": type_right,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, np.intp, np.int32])
    return func

def get_mulscalar_function(src_type, pitch = True):
    type_src = dtype_to_ctype(src_type)

    name = "mulscalar"
    operation = "*"
    
    if pitch:
        func = func_compile(name, pitch_left_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, src_type.type])
    else:
        func = func_compile(name, non_pitch_left_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, src_type.type, np.int32])
    return func


def get_divarray_function(left_dtype, right_dtype, rslt_dtype, pitch = True):
    type_left = dtype_to_ctype(left_dtype)
    type_right = dtype_to_ctype(right_dtype)
    type_rslt = dtype_to_ctype(rslt_dtype)

    name = "divarray"
    operation = "/"

    if pitch:
        func = func_compile(name, pitch_array_op_template % {"name": name,
                   "dest_type": type_rslt,
                   "left_type": type_left,
                   "right_type": type_right,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, np.intp, np.int32])
    else:
        func = func_compile(name, non_pitch_array_op_template % {"name": name,
                   "dest_type": type_rslt,
                   "left_type": type_left,
                   "right_type": type_right,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, np.intp, np.int32])
    return func

def get_divscalar_function(src_type, pitch = True):
    type_src = dtype_to_ctype(src_type)

    name = "divscalar"
    operation = "/"
    
    if pitch:
        func = func_compile(name, pitch_left_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, src_type.type])
    else:
        func = func_compile(name, non_pitch_left_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, src_type.type, np.int32])
    return func

def get_scalardiv_function(src_type, pitch = True):
    type_src = dtype_to_ctype(src_type)

    name = "scalardiv"
    operation = "/"
    
    if pitch:
        func = func_compile(name, pitch_right_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.int32, np.int32, np.intp, np.int32, np.intp, np.int32, src_type.type])
    else:
        func = func_compile(name, non_pitch_right_scalar_op_template % {"name": name,
                   "dest_type": type_src,
                   "operation": operation,
                   })
        func.prepare([np.intp, np.intp, src_type.type, np.int32])
    return func



transpose_template = """
    #include <pycuda-complex.hpp>
    #define TILE_DIM 32
    #define BLOCK_ROWS 8
    
    __global__ void %(name)s(const int M, const int N, %(type)s *odata, const int ldo, const %(type)s *idata, const int ldi)
    {
        __shared__ %(type)s tile[TILE_DIM][TILE_DIM+1];
        int xIndex, yIndex, index_in, index_out;
        int MM = ((M-1) >> 5) + 1;
        int NN = ((N-1) >> 5) + 1;

        for(int i = blockIdx.x; i < MM * NN; i += gridDim.x)
        {
            xIndex = (i %% NN) * TILE_DIM + threadIdx.x;
            yIndex = (i / NN) * TILE_DIM + threadIdx.y;
            index_in = xIndex + ldi * yIndex;

            if(xIndex < N)
            {
                for(int j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                {
                    if(yIndex + j < M)
                    {
                        tile[threadIdx.y+j][threadIdx.x] = (idata[index_in+j*ldi]);
                    }
                }
            }
            __syncthreads();

            xIndex = (i / NN) * TILE_DIM + threadIdx.x;
            yIndex = (i %% NN) * TILE_DIM + threadIdx.y;
            index_out = xIndex + ldo * yIndex;

            if(xIndex < M)
            {
                for(int j = 0; j < TILE_DIM; j+=BLOCK_ROWS)
                {
                    if(yIndex + j < N)
                    {
                        odata[index_out + j * ldo] = %(operation)s(tile[threadIdx.x][threadIdx.y+j]);
                    }
                }
            }
            __syncthreads();
        
        }
    }
    """





"""templates"""

pitch_template = """
            #include <pycuda-complex.hpp>
            #include <cuComplex.h>

            __global__ void %(name)s (const int M, const int N, %(dest_type)s *dest, const int ld_dest, const %(src_type)s *src, const int ld_src)
            {
                //M is the number of rows, N is the number of columns
                const int tid = threadIdx.x;
                const int sid = threadIdx.y + blockDim.y * blockIdx.x;
                const int total = gridDim.x * blockDim.y;

                int m, n, idx;
                %(src_type)s tmp;
                int segment_per_row = ((N - 1) >> 5) + 1;
                int total_segments = M * segment_per_row;
                
                for(int i = sid; i < total_segments; i+=total)
                {
                    m = i / segment_per_row;
                    n = i %% segment_per_row;
                    idx = (n<<5) + tid;
                    if(idx < N)
                    {
                        tmp = src[m * ld_src + idx];
                        dest[m * ld_dest + idx] = %(operation)s (tmp);
                    }
                }
            }
            """
""" launching MULTIPROCESSOR_COUNT*6 blocks of (32,8,1), M: number of rows, N: number of columns, ld: leading dimension entries(aasumed to be row major)"""


            
non_pitch_template = """
            #include <pycuda-complex.hpp>
            #include <cuComplex.h>
            
            __global__ void %(name)s (%(dest_type)s *dest, const %(src_type)s *src, const int N)
            {
                const int totalthreads = blockDim.x * gridDim.x;
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                %(src_type)s tmp;

                for (int i = tid; i < N; i += totalthreads)
                {
                    tmp = src[i];
                    dest[i] =  %(operation)s (tmp);
                }
            }
            """
""" launching MULTIPROCESSOR_COUNT*6 blocks of (256,1,1), N = totalsize"""


reshape_template = """
            #include <pycuda-complex.hpp>
            #include <cuComplex.h>
            __global__ void %(name)s(const int Msrc, const int Nsrc, const int Mdest, const int Ndest, %(dest_type)s *dest, const int ld_dest, const %(src_type)s *src, const int ld_src)
            {
                const int totalthreads = blockDim.x * gridDim.x;
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;
                
                %(src_type)s tmp;

                for (int i = tid; i < Nsrc * Msrc; i += totalthreads)
                {
                    tmp = src[i / Nsrc * ld_src + i %% Nsrc];
                    dest[i / Ndest * ld_dest + i %% Ndest] = %(operation)s (tmp);
                }
            }
            """
""" launching MULTIPROCESSOR_COUNT*6 blocks of (256,1,1), M: number of rows, N: number of columns, ld: leading dimension entries(aasumed to be row major)"""

irregular_pitch_template = """
            #include <pycuda-complex.hpp>
            #include <cuComplex.h>
            __global__ void %(name)s(const int M, const int N, %(dest_type)s *dest, const int ld_dest, const %(src_type)s *src, const int ld_src)
            {
                const int totalthreads = blockDim.x * gridDim.x;
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;
                
                %(src_type)s tmp;
                int m,n;

                for (int i = tid; i < N * M; i += totalthreads)
                {
                    m = i / N;
                    n = i %% N;
                    tmp = src[m * ld_src + n];
                    dest[m * ld_dest + n] =  %(operation)s (tmp);
                }
            }
            """
""" launching MULTIPROCESSOR_COUNT*6 blocks of (256,1,1), M: number of rows, N: number of columns, ld: leading dimension entries(aasumed to be row major)"""


pitch_array_op_template = """
    #include <pycuda-complex.hpp>
    
    extern "C++" {
	namespace pycuda{

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator+(const complex<_Tp1>& __z, const _Tp2& __x)
	{return complex<double>(__z._M_re + __x, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator+(const _Tp2& __x, const complex<_Tp1>& __z)
	{return complex<double>(__x + __z._M_re, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator-(const complex<_Tp1>& __z, const _Tp2& __x)
	{return complex<double>(__z._M_re - __x, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator-(const _Tp2& __x, const complex<_Tp1>& __z)
	{return complex<double>(__x - __z._M_re, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator*(const complex<_Tp1>& __z, const _Tp2& __x)
	{return (complex<double>)__z * __x;}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator*(const _Tp2& __x, const complex<_Tp1>& __z)
	{return (complex<double>)__z * __x;}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator/(const complex<_Tp1>& __z, const _Tp2& __x)
	{return (complex<double>)__z / __x;}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator/(const _Tp2& __x, const complex<_Tp1>& __z)
	{return __x / (complex<double>)__z;}

	}
}
    
    
    __global__ void %(name)s(const int M, const int N, %(dest_type)s *dest, const int ld_dest, const %(left_type)s *left, const int ld_left, const %(right_type)s *right, const int ld_right)
    {
        //M is the number of rows, N is the number of columns
        const int tid = threadIdx.x;
        const int sid = threadIdx.y + blockDim.y * blockIdx.x;
        const int total = gridDim.x * blockDim.y;

        int m, n, idx;
        %(left_type)s tmp_left;
        %(right_type)s tmp_right;
        int segment_per_row = ((N - 1) >> 5) + 1;
        int total_segments = M * segment_per_row;
        
        for(int i = sid; i < total_segments; i+=total)
        {
            m = i / segment_per_row;
            n = i %% segment_per_row;
            idx = (n<<5) + tid;
            if(idx < N)
            {
                tmp_left = left[m * ld_left + idx];
                tmp_right = right[m * ld_right + idx];
                dest[m * ld_dest + idx] = (tmp_left) %(operation)s (tmp_right);
            }
        }
    }
    """

non_pitch_array_op_template = """
    #include <pycuda-complex.hpp>
    
    
    extern "C++" {
	namespace pycuda{

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator+(const complex<_Tp1>& __z, const _Tp2& __x)
	{return complex<double>(__z._M_re + __x, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator+(const _Tp2& __x, const complex<_Tp1>& __z)
	{return complex<double>(__x + __z._M_re, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator-(const complex<_Tp1>& __z, const _Tp2& __x)
	{return complex<double>(__z._M_re - __x, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator-(const _Tp2& __x, const complex<_Tp1>& __z)
	{return complex<double>(__x - __z._M_re, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator*(const complex<_Tp1>& __z, const _Tp2& __x)
	{return (complex<double>)__z * __x;}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator*(const _Tp2& __x, const complex<_Tp1>& __z)
	{return (complex<double>)__z * __x;}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator/(const complex<_Tp1>& __z, const _Tp2& __x)
	{return (complex<double>)__z / __x;}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator/(const _Tp2& __x, const complex<_Tp1>& __z)
	{return __x / (complex<double>)__z;}

	}
}

    __global__ void %(name)s(%(dest_type)s *dest, const %(left_type)s *left, const %(right_type)s *right, const int N)
    {
        const int totalthreads = blockDim.x * gridDim.x;
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;

        %(left_type)s tmp_left;
        %(right_type)s tmp_right;

        for (int i = tid; i < N; i += totalthreads)
        {
            tmp_left = left[i];
            tmp_right = right[i];
            dest[i] =  (tmp_left) %(operation)s (tmp_right);
        }

    }

    """

pitch_left_scalar_op_template = """
    #include <pycuda-complex.hpp>
    __global__ void %(name)s(const int M, const int N, %(dest_type)s *dest, const int ld_dest, const %(dest_type)s *left, const int ld_left, const %(dest_type)s right)
    {
        //M is the number of rows, N is the number of columns
        const int tid = threadIdx.x;
        const int sid = threadIdx.y + blockDim.y * blockIdx.x;
        const int total = gridDim.x * blockDim.y;

        int m, n, idx;
        %(dest_type)s tmp_left;
        int segment_per_row = ((N - 1) >> 5) + 1;
        int total_segments = M * segment_per_row;
        
        for(int i = sid; i < total_segments; i+=total)
        {
            m = i / segment_per_row;
            n = i %% segment_per_row;
            idx = (n<<5) + tid;
            if(idx < N)
            {
                tmp_left = left[m * ld_left + idx];
                dest[m * ld_dest + idx] = (tmp_left) %(operation)s (right);
            }
        }
    }
    """

non_pitch_left_scalar_op_template = """
    #include <pycuda-complex.hpp>
    __global__ void %(name)s(%(dest_type)s *dest, const %(dest_type)s *left, const %(dest_type)s right, const int N)
    {
        const int totalthreads = blockDim.x * gridDim.x;
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;

        %(dest_type)s tmp_left;

        for (int i = tid; i < N; i += totalthreads)
        {
            tmp_left = left[i];
            dest[i] =  (tmp_left) %(operation)s (right);
        }

    }

    """

pitch_right_scalar_op_template = """
    #include <pycuda-complex.hpp>
    __global__ void %(name)s(const int M, const int N, %(dest_type)s *dest, const int ld_dest, const %(dest_type)s *left, const int ld_left, const %(dest_type)s right)
    {
        //M is the number of rows, N is the number of columns
        const int tid = threadIdx.x;
        const int sid = threadIdx.y + blockDim.y * blockIdx.x;
        const int total = gridDim.x * blockDim.y;

        int m, n, idx;
        %(dest_type)s tmp_left;
        int segment_per_row = ((N - 1) >> 5) + 1;
        int total_segments = M * segment_per_row;
        
        for(int i = sid; i < total_segments; i+=total)
        {
            m = i / segment_per_row;
            n = i %% segment_per_row;
            idx = (n<<5) + tid;
            if(idx < N)
            {
                tmp_left = left[m * ld_left + idx];
                dest[m * ld_dest + idx] = (right) %(operation)s (tmp_left);
            }
        }
    }
    """

non_pitch_right_scalar_op_template = """
    #include <pycuda-complex.hpp>
    __global__ void %(name)s(%(dest_type)s *dest, const %(dest_type)s *left, const %(dest_type)s right, const int N)
    {
        const int totalthreads = blockDim.x * gridDim.x;
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;

        %(dest_type)s tmp_left;

        for (int i = tid; i < N; i += totalthreads)
        {
            tmp_left = left[i];
            dest[i] =  (right) %(operation)s (tmp_left);
        }

    }

    """

fill_pitch_template = """
            #include <pycuda-complex.hpp>

            __global__ void %(name)s (const int M, const int N, %(type_dst)s *dst, const int ld_dst, %(type_dst)s value)
            {
                //M is the number of rows, N is the number of columns
                const int tid = threadIdx.x;
                const int sid = threadIdx.y + blockDim.y * blockIdx.x;
                const int total = gridDim.x * blockDim.y;

                int m, n;
                int segment_per_row = ((N - 1) >> 5) + 1;
                int total_segments = M * segment_per_row;
                
                for(int i = sid; i < total_segments; i+=total)
                {
                    m = i / segment_per_row;
                    n = i %% segment_per_row;
                    dst[m * ld_dst + (n<<5) + tid] = value; 
                }
            }
            """
""" launching MULTIPROCESSOR_COUNT*6 blocks of (32,8,1), M: number of rows, N: number of columns, ld: leading dimension entries(aasumed to be row major)"""

fill_nonpitch_template = """
            #include <pycuda-complex.hpp>

            __global__ void %(name)s (const int M,  %(type_dst)s *dst, %(type_dst)s value)
            {
                //M is the number of rows, N is the number of columns
                const int tid = threadIdx.x + blockDim.x * blockIdx.x;
                const int total = gridDim.x * blockDim.x;

                
                
                for(int i = tid; i < M; i+=total)
                {
                    dst[i] = value; 
                }
            }
            """
