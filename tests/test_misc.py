#!/usr/bin/env python

from unittest import main, TestCase

import nk.tools.misc as misc

class test_misc(TestCase):
    def test_no_duplicates_consecutive_in_order(self):
        result = misc.renumber_in_order([0, 1, 2])
        self.assertSequenceEqual(result,
                                 [0, 1, 2])

    def test_no_duplicates_nonconsecutive_in_order(self):
        result = misc.renumber_in_order([0, 2, 4])
        self.assertSequenceEqual(result,
                                 [0, 1, 2])

    def test_duplicates_consecutive_in_order(self):
        result = misc.renumber_in_order([0, 0, 1, 2, 2, 3])
        self.assertSequenceEqual(result,
                                 [0, 0, 1, 2, 2, 3])

    def test_duplicates_nonconsecutive_in_order(self):
        result = misc.renumber_in_order([0, 0, 2, 3, 3, 5])
        self.assertSequenceEqual(result,
                                 [0, 0, 1, 2, 2, 3])

    def test_duplicates_nonconsecutive_out_of_order(self):
        result = misc.renumber_in_order([0, 3, 1, 1, 2, 3])
        self.assertSequenceEqual(result,
                                 [0, 1, 2, 2, 3, 1])

if __name__ == '__main__':
    main()

