#!/usr/bin/env python

"""
Generate unique consecutive identifiers.
"""

import itertools, re

global _count
_count = itertools.count()

def uuid(n=5):
    """
    Generate a UUID with the specified length.
    """

    global _count
    c = _count.next()
    if len(str(c)) > n:
        raise ValueError('UUID width exceeded')
    return re.sub('\s', '0', ('{c:%i}' % n).format(c=c))
