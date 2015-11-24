#!/usr/bin/env python

import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name is None:
            print('Starting timer'.format(self.name))
        else:
            print('Starting {}'.format(self.name))

    def __exit__(self, type, value, traceback):
        if self.name is None:
            print('Elapsed time: {0:.2f}s'
                  .format(time.time() - self.tstart))
        else:
            print('Elapsed time for {1}: {0:.2f}s'
                  .format(time.time() - self.tstart, self.name))


def main():
    with Timer():
        time.sleep(1)
    with Timer('Test'):
        time.sleep(2)

if __name__ == '__main__':
    main()
