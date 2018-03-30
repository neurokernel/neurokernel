import tempfile
import os
import inspect
import subprocess
import dill
import re

from neurokernel.mixins import LoggerMixin

def mpi_run(func, targets=None, delete_tempfile=True, log=True,
            log_screen=True, log_file_name='neurokernel.log'):
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
    This will only work when the target class is in a library (like LPU),
        if the source for the class is in the source file it will not work. This
        can be extended to be more general if anyone has a use for it.
    Does not seem to work with openmpi version 2
    func should not import neurokernel.mpi_relaunch
    All modules and variables used must be imported or defined within func
    Returns the stdout from the function run under 'mpiexec -np 1 python {tmp_file_name}'
    """

    l = LoggerMixin("mpi_run()",log_on=log)
    
    if callable(func):
        func_text = inspect.getsource(func)
        # Make a feeble attempt at fixing indentation. Will work for a nested function
        # that takes no args, not a member function that expects (self) or a class
        func_text = "\n" + re.sub(r"(^\s+)def ","def ",func_text) + "\n"
        func_name = func.__name__
    else: 
        func_text = "\n" + func + "\n"
        func_name = re.search('def *(.*)\(\):', func_text).group(1)

    target_text  = "\n"
    
    if targets:
        for t in targets:
            target_text +=  "\n" + inspect.getsource(t) + "\n"

    main_code  = "\n"
    main_code += "\nif __name__ == \"__main__\":"
    main_code += "\n   import neurokernel.mpi as mpi"
    main_code += "\n   from neurokernel.mixins import LoggerMixin"
    main_code += "\n   from mpi4py import MPI"

    if log:
        main_code += "\n   mpi.setup_logger(screen=%s, file_name=\"%s\"," % (log_screen, log_file_name)
        main_code += "\n                    mpi_comm=MPI.COMM_WORLD, multiline=True)"

    main_code += "\n   l = LoggerMixin(\"%s\",%s)" % (func_name,str(log))
    main_code += "\n   try:"
    main_code += "\n      %s()" % func_name
    main_code += "\n      print(\"MPI_RUN_SUCCESS: %s\")" % func_name
    main_code += "\n      l.log_info(\"MPI_RUN_SUCCESS: %s\")" % func_name
    main_code += "\n   except Exception as e:"
    main_code += "\n      print(\"MPI_RUN_FAILURE: %s\")" % func_name
    main_code += "\n      l.log_error(\"MPI_RUN_FAILURE: %s\")" % func_name
    main_code += "\n      print(e)"
    main_code += "\n"

    try:
        #Write code for the function to a temp file
        temp = tempfile.NamedTemporaryFile(delete = delete_tempfile)
        temp.write(target_text)
        temp.write(func_text)
        temp.write(main_code)
        temp.flush()
        
        #Execute the code
        env = os.environ.copy()
        command = ["mpiexec", "-np", "1", "python", temp.name]
        l.log_info("Calling: " + " ".join(command))
        out = subprocess.check_output(command, env = env)

    except Exception as e:
        l.log_error(str(e))
        raise

    finally:    
        #Closing the temp file closes and deletes it
        temp.close()
    
    #Return the output
    if "MPI_RUN_FAILURE" in out:
        raise RuntimeError(out)

    return str(out)


def mpi_run_manager(man, steps, targets=None, delete_tempfile=True, log=True,
                    log_screen=True, log_file_name='neurokernel.log'):
    """
    Run the manager with mpiexec.
    """

    l = LoggerMixin("mpi_run_manager()",log_on=log)

    #Write a function that loads and runs the Manager
    func_code  = "\ndef MPI_Function():"
    func_code += "\n    import dill"
    func_code += "\n    f = open(\"%s\",\"rb\")"
    func_code += "\n    man = dill.load(f)"
    func_code += "\n    man.spawn()" 
    func_code += "\n    man.start(steps=%i)"
    func_code += "\n    man.wait()"

    try:
        #Store the Manager in a temporary file
        temp = tempfile.NamedTemporaryFile(delete = delete_tempfile)
        dill.dump(man, temp)
        temp.flush()
        
        #Run the function using mpiexec
        out = mpi_run(func_code % (temp.name,steps), targets, 
                      delete_tempfile=delete_tempfile, log=log,
                      log_screen=log_screen, log_file_name=log_file_name)

    except Exception as e:
        l.log_error(str(e))
        raise

    finally:    
        #Closing the temp file closes and deletes it
        temp.close()
    
    #Return the output
    return str(out)


#Basic sanity checks
if __name__ == "__main__":

    from tools.logging import setup_logger
    setup_logger(screen=True, file_name='neurokernel.log', multiline=True)

    def _test_success(): 
        print("HELLO WORLD")

    def _test_fail(): 
        open("RANDOM_FILE", "r").read() 

    print("This should succeed:")
    print(mpi_run(_test_success))

    print("This should also succeed:")
    code  = "\ndef func():"
    code += "\n   print(\"HELLO AGAIN\")"
    print(mpi_run(code))

    print("This should fail:")
    print(mpi_run(_test_fail))

