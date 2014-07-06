import tables
import numpy as np
try:
    import pycuda.gpuarray as garray
    PYCUDA = True
except ImportError:
    PYCUDA = False


def file_attr(filename):
    """
    Read attribute of the file
    returns the shape of the array and the dtype
    """
    h5file = tables.openFile(filename)
    n = h5file.root._v_nchildren
    if n == 1:
        c = h5file.root._f_listNodes()[0]
        dtype = c.dtype
        shape = c.shape
    elif n == 2:
        if h5file.root.real.dtype == np.float32:
            dtype = np.dtype(np.complex64)
        elif h5file.root.real.dtype == np.float64:
            dtype = np.dtype(np.complex128)
        else:
            raise TypeError("file dtype not supported")
        shape = h5file.root.real.shape
    h5file.close()
    return shape, dtype
    

def write_memory_to_file(A, filename, mode = 'w',
                         title = 'test', complevel = None,
                         verbose = True):
    """
    write memory to a h5 file
    h5 file contains root.real and root.imag(if A complex)
    best for transfer data with Matlab
    
    A: a ndarray, GPUArray or PitchArray
    filename: name of file to store
    mode: 'w' to start a new file
          'a' to append, leading dimension of A must be the
           same as the existing file
    
    file can be read by read_file or in matlab using h5read.m
    """
    h5file = tables.openFile(filename, mode, title)
    
    if complevel is not None:
        filters = tables.Filters(complevel=complevel, complib='zlib')
    else:
        filters = None

    if (A.dtype == np.float32) or (A.dtype == np.complex64):
        tb = tables.Float32Atom
    elif (A.dtype == np.float64) or (A.dtype == np.complex128):
        tb = tables.Float64Atom
    elif A.dtype == np.int32:
        tb = tables.Int32Atom
    elif A.dtype == np.int64:
        tb = tables.Int64Atom
    else:
        TypeError("Write file error: unkown input dtype")

    if PYCUDA:
        if A.__class__.__name__ in ["GPUArray", "PitchArray"]:
            B = A.get()
        elif A.__class__.__name__ == "ndarray":
            B = A
        else:
            raise TypeError("Write file error: unkown input")
    else:
        if A.__class__.__name__ == "ndarray":
            B = A
        else:
            raise TypeError("Write file error: unkown input")

    shape = list(B.shape)
    shape[0] = 0
    
    if mode == 'w':
        if np.iscomplexobj(B):
            h5file.createEArray("/","real", tb(),
                                tuple(shape), filters = filters)
            h5file.createEArray("/","imag", tb(),
                                tuple(shape), filters = filters)
        else:
            h5file.createEArray("/","real", tb(),
                                tuple(shape), filters = filters)

    if np.iscomplexobj(B):
        h5file.root.real.append(B.real)
        h5file.root.imag.append(B.imag)
    else:
        h5file.root.real.append(B)

    h5file.close()
    if verbose:
        if mode == 'w':
            print "file %s created" % (filename)
        else:
            print "file %s attached" % (filename)


def write_array(A, filename, mode = 'w', title = 'test',
                complevel = None, verbose = True):
    """
    write memory to a h5 file
    h5 file contains root.arrat(A real or complex)
    
    A: a ndarray, GPUArray or PitchArray
    filename: name of file to store
    mode: 'w' to start a new file
          'a' to append, leading dimension of A must
           be the same as the existing file
    
    file can be read by read_array in python
    """

    h5file = tables.openFile(filename, mode, title)

    if complevel is not None:
        filters = tables.Filters(complevel=complevel, complib='zlib')
    else:
        filters = None
    
    if (A.dtype == np.float32):
        tb = tables.Float32Atom
    elif (A.dtype == np.float64):
        tb = tables.Float64Atom
    elif (A.dtype == np.complex64) or (A.dtype == np.complex128):
        tb = tables.ComplexAtom
    elif A.dtype == np.int32:
        tb = tables.Int32Atom
    elif A.dtype == np.int64:
        tb = tables.Int64Atom
    else:
        TypeError("Write file error: unkown input dtype")

    if PYCUDA:
        if A.__class__.__name__ in ["GPUArray", "PitchArray"]:
            B = A.get()
        elif A.__class__.__name__ == "ndarray":
            B = A
        else:
            raise TypeError("Write file error: unkown input")
    else:
        if A.__class__.__name__ == "ndarray":
            B = A
        else:
            raise TypeError("Write file error: unkown input")

    shape = list(B.shape)
    shape[0] = 0
    
    if mode == 'w':
        if (A.dtype == np.complex64):
            h5file.createEArray("/","array", tb(8),
                                tuple(shape), filters = filters)
        elif (A.dtype == np.complex128):
            h5file.createEArray("/","array", tb(16),
                                tuple(shape), filters = filters)
        else:
            h5file.createEArray("/","array", tb(),
                                tuple(shape), filters = filters)
    h5file.root.array.append(B)
    h5file.close()
    if verbose:
        if mode == 'w':
            print "file %s created" % (filename)
        else:
            print "file %s attached" % (filename)


def read_file(filename, start = None, stop = None, step = 1):
    """
    read a h5 file generated by write_memory_to_file or
    in Matlab using h5write.m
    returns a ndarray
    """
    h5file = tables.openFile(filename, "r")
    n = h5file.root._v_nchildren

    if n == 1:
        a = h5file.root.real.read(start = start, stop = stop, step = step)
    elif n == 2:
        r = h5file.root.real.read(start = start, stop = stop, step = step)
        i = h5file.root.imag.read(start = start, stop = stop, step = step)
        cl = r.dtype
        #a = h5file.root.real.read() + 
        #    (np.array(1j).astype(np.complex64))*h5file.root.imag.read()
        a = r + (np.array(1j).astype(np.complex64))*i
    h5file.close()    
    return a


def read_array(filename, start = None, stop = None, step = 1):
    """
    read a h5 file generated by write_array
    returns a ndarray
    """
    h5file = tables.openFile(filename, "r")
    n = h5file.root._v_nchildren
        
    a = h5file.root.array.read(start = start, stop = stop, step = step)
    h5file.close()
    return a


def read(filename, start = None, stop = None, step = 1):
    """
    Read an h5file written either by Matlab or python
    """
    h5file = tables.openFile(filename, "r")
    n = h5file.root._v_nchildren

    if n == 1:
        c = h5file.root._f_listNodes()[0]
        a = c.read(start = start, stop = stop, step = step)
    elif n == 2:
        r = h5file.root.real.read(start = start, stop = stop, step = step)
        i = h5file.root.imag.read(start = start, stop = stop, step = step)
        a = r + 1j*i
    h5file.close()    
    return a

    
    
    
