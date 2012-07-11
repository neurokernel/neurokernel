#!/usr/bin/env python

"""
Generate unique consecutive identifiers.

Notes
-----
Generated identifiers are only unique within the same process.
"""

import itertools, re

global _count
_count = itertools.count()

def uid(n=5):
    """
    Generate a UID with the specified length.
    """

    global _count
    c = _count.next()
    if len(str(c)) > n:
        raise ValueError('UID width exceeded')
    return re.sub('\s', '0', ('{c:%i}' % n).format(c=c))
