#!/usr/bin/env python

"""
Given a matrix of integers x that represent the number of connections between row
i and column j (where x[i, i] = 0) and starting with the indices i and j for 
which x[i, j]+x[j, i] is maximal over all (i, j), order the pairs of indices 
(i, j) such that each x[i_curr, j_curr]+x[j_curr, i_curr] increases the total
sum of x[i, j]+x[j, i] over already seen indices by the maximum amount possible
for the remaining index pairs.
"""

import itertools

import numpy as np

def interleave(a, b):
    """
    Interleave elements of two lists of equal length.
    """

    return list(itertools.chain.from_iterable(itertools.izip(a, b)))

def get_index_order(x):
    x = np.asarray(x)
    assert x.shape[0] == x.shape[1]
    N = x.shape[0]

    # Find indices (i, j) of elements x[i, j], x[j, i] such that x[i, j]+x[j, i] is
    # maximized (elements x[i, i] are ignored):
    first = [(i, j) for i, j in itertools.product(xrange(N), xrange(N)) if i > j]
    second = [(j, i) for i, j in itertools.product(xrange(N), xrange(N)) if i > j]

    inds = interleave(first, second)
    x_ordered = [x[i, j] for i, j in inds]
    x_ordered_summed = [a+b for a, b in zip(x_ordered[::2], x_ordered[1::2])]
    i_ordered_summed_max = np.argmax(x_ordered_summed)
    i_ordered_max = i_ordered_summed_max*2
    ind_max = inds[i_ordered_max]

    # Indices already added:
    added_inds = [ind_max[0], ind_max[1]]

    # Remaining indices to consider:
    remaining_inds = range(N)
    for ind in added_inds:
        remaining_inds.remove(ind)

    while remaining_inds:

        # For each remaining index i and each index j already added, compute
        # increase due to adding values x[i, j]+x[j, i]:
        sums = []
        for i in remaining_inds:
            s = sum([x[i, j]+x[j, i] for j in added_inds])
            sums.append(s)

        # Add the index corresponding to the maximum increase to added_inds and
        # remove it from remaining_inds:
        i_max = np.argmax(sums)
        added_inds.append(remaining_inds[i_max])
        del remaining_inds[i_max]
    return np.asarray(added_inds)

if __name__ == '__main__':
    # Create example matrix:
    N = 10
    x = np.random.randint(0, 20, (N, N))
    for i in xrange(N):
        x[i, i] = 0

    add_inds = get_index_order(x)
