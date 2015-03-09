from allglobalvars import allglobalvars

import inspect
import random
from unittest import main, TestCase

def func1(x):
    return x+random.random()

from numbers import Integral

def func2(x):
    return isinstance(x, Integral)

class test_allglobalvars(TestCase):
    def test_func(self):
        r = allglobalvars(func1)
        assert r.has_key('random') and inspect.ismodule(r['random'])

    def test_func_imported(self):
        r = allglobalvars(func2)
        assert len(r.keys()) == 1 and \
            r.has_key('Integral') and inspect.isclass(r['Integral'])

    def test_class_member(self):
        class Foo(object):
            def foo(self, x):
                return random.random()+x

        r = allglobalvars(Foo)
        assert len(r.keys()) == 1 and \
            r.has_key('random') and inspect.ismodule(r['random'])

    def test_class_instance_member(self):
        class Foo(object):
            def foo(self, x):
                return random.random()+x

        r = allglobalvars(Foo())
        assert len(r.keys()) == 1 and \
            r.has_key('random') and inspect.ismodule(r['random'])

    def test_class_member_calls_func(self):
        class Foo(object):
            def foo(self, x):
                return func1(x)

        r = allglobalvars(Foo)
        assert len(r.keys()) == 2 and \
            r.has_key('random') and inspect.ismodule(r['random']) and \
            r.has_key('func1') and inspect.isfunction(r['func1'])

    def test_class_child(self):
        class Foo(object):
            def foo(self, x):
                return random.random()+x

        class Bar(Foo):
            def foo(self, x):
                return super(Bar, self).foo(x)

        r = allglobalvars(Bar)
        assert len(r.keys()) == 3 and \
            r.has_key('random') and inspect.ismodule(r['random']) and \
            r.has_key('Foo') and inspect.isclass(r['Foo']) and \
            r.has_key('Bar') and inspect.isclass(r['Bar'])

    def test_class_in_func(self):
        class Foo(object):
            def foo(self, x):
                return random.random()+x

        def bar(x):
            return Foo().foo(x)

        r = allglobalvars(bar)
        assert len(r.keys()) == 2 and \
            r.has_key('random') and inspect.ismodule(r['random'])
        
if __name__ == '__main__':
    main()
