#!/usr/bin/env python

"""
How to find globals accessed by a Python object.
"""

import inspect

import numpy as np

def allglobalvars(x):
    """
    Find all globals accessed by an object.
    """

    # Define internal recursively called function to enable tracking of the
    # recursion level:
    def recursive(x, seen=set(), level=0):

        # Get locals of scope in which allglobalvars() was invoked; the
        # following code ensures that the `locals_dict` contains the locals from
        # the top scope regardless of the recursion level:
        local_dict = inspect.currentframe(level+2).f_locals
        level += 1

        results = {}
        if inspect.isbuiltin(x) or type(x) == np.ufunc:        
            return {x.__name__: x}
        elif inspect.isroutine(x):

            # Get symbols and globals accessed by function or method; we need to
            # check co_freevars for defined symbols that are not in co_names:
            if inspect.isfunction(x):
                name_list = x.func_code.co_names+x.func_code.co_freevars
                global_dict = x.func_globals
            elif inspect.ismethod(x):
                name_list = x.im_func.func_code.co_names+x.im_func.func_code.co_freevars
                global_dict = x.im_func.func_globals
            else:
                raise ValueError('invalid input')

            for name in name_list:
                if name in seen:
                    pass
                elif name in global_dict:

                    # If an accessed symbol is a global that can be imported as-is, 
                    # include it in the dict of globals; if it is a method of a class,
                    # recurse into it:
                    results[name] = global_dict[name]
                    if not inspect.ismodule(global_dict[name]) and \
                       hasattr(global_dict[name], '__module__') and \
                       global_dict[name].__module__ == '__main__':
                        results.update(recursive(global_dict[name],
                                                 set([name]).union(seen), level))
                elif name in local_dict:
                    results[name] = local_dict[name]
                    if not inspect.ismodule(local_dict[name]) and \
                       hasattr(local_dict[name], '__module__') and \
                       local_dict[name].__module__ == '__main__':
                        results.update(recursive(local_dict[name],
                                                 set([name]).union(seen), level))
                else:
                    # Check if symbol is the name is an attribute of a global (i.e.,
                    # can be imported):
                    for r in results.keys():
                        if hasattr(results[r], name) and \
                           inspect.ismodule(getattr(results[r], name)):
                            results[r+'.'+name] = getattr(results[r], name)

        else:
            # Include globals of a class' parents:
            if inspect.isclass(x):

                # Parent classes other than object should be included in a class'
                # globals:
                for b in x.__bases__:
                    if b is not object:
                        results[b.__name__] = b
                    results.update(recursive(b, set([b.__name__]).union(seen), level))

            # Recurse into class/object methods:
            for f in inspect.getmembers(x, predicate=inspect.ismethod): 
                results.update(recursive(f[1], set([f[1].__name__]).union(seen), level))
        return results
    return recursive(x)

if __name__ == '__main__':

    # Demo of usage:
    import numpy as np
    class Foo(object):
        def bar(self, x):
            return np.linalg.svd(x)

    print allglobalvars(Foo)
    print '---'

    import math
    f = lambda x: math.lgamma(x)
    print allglobalvars(f)

    print '---'
    import pandas
    print allglobalvars(pandas.DataFrame)
