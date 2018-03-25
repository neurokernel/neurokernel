import tempfile
import os
import inspect
import subprocess

def mpi_run(func):
    """
    Run a function with mpiexec.
    
    Implemented as a fix to 'import neurokernel.mpi_relaunch', which does not
    work within notebooks. Writes the source code for a function to a temporary
    file and then runs the temporary file using mpiexec. Returns the stdout of
    from the function along with a string indicating whether or not the function
    executed properly.

    Parameters
    ----------
    func : function
        Function to be executed with mpiexec. All imports and variables used
        must be imported or defined within the function.
    
    Returns
    -------
    output : str
        The stdout from the function run with mpiexec cast to a string.

    Usage
    -----
    Does not seem to work with openmpi version 2
    func should not import neurokernel.mpi_relaunch
    All modules and variables used must be imported or defined within func
    Returns the stdout from the function run under 'mpiexec -np 1 python {tmp_file_name}'
    """
    
    #Create a temporary file
    temp, filename = tempfile.mkstemp()
    
    #Write code from func into the file
    os.write(temp, inspect.getsource(func))
    
    #Add a main
    os.write(temp, "\nif __name__ == \"__main__\":")
    os.write(temp, "\n   try:")
    os.write(temp, "\n      %s()" % func.__name__)
    os.write(temp, "\n      print(\"%s executed successfully\")" % func.__name__)
    os.write(temp, "\n   except Exception as e:")
    os.write(temp, "\n      print(\"%s failed\")" % func.__name__) 
    os.write(temp, "\n      print(e)")
    
    #Extract the current environment
    env = os.environ.copy()
    
    #Execute the code
    out = subprocess.check_output(["mpiexec", "-np", "1", "python", filename], env = env)
    
    #Close the file
    os.close(temp)
    
    #Return the output
    return str(out)

def _test_success(): 
  print("HELLO WORLD")

def _test_fail(): 
  open("RANDOM_FILE", "r").read() 

#A test for the function
if __name__ == "__main__":

   print("This should succeed:")
   print(mpi_run(_test_success))

   print("This should fail:")
   print(mpi_run(_test_fail))
