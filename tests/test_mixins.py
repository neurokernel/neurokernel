#!/usr/bin/env python 

import sys
from unittest import main, TestCase

import twiggy
from twiggy.lib.converter import ConversionTable

import neurokernel.mixins as mixins

# Create formatter and filter to produce predictable output that can be tested:
conv = ConversionTable()
conv.add('name', str, '{1}'.format)
conv.add('level', str, '{1}'.format)
conv.add('message', str, '{1}'.format)
conv.genericValue = str
conv.genericItem = '{0}={1}'.format
conv.aggregate = ':'.join
fmt = twiggy.formats.LineFormat(conversion=conv)

def filt(msg):
    del msg.fields['time']
    return msg

class test_loggermixin(TestCase):
    def setUp(self):
        output = twiggy.outputs.StreamOutput(format=fmt, stream=sys.stdout)
        twiggy.emitters['*'] = twiggy.filters.Emitter(twiggy.levels.DEBUG, filt, output)
        self.lm = mixins.LoggerMixin('log')

    def test_methods(self):
        self.lm.log_debug('abc')
        self.lm.log_info('abc')
        self.lm.log_warning('abc')
        self.lm.log_error('abc')
        self.lm.log_critical('abc')

        # All output to stdout is buffered within a single test before emission:
        self.assertEquals(sys.stdout.getvalue().strip(),
                          'log:DEBUG|abc\n'
                          'log:INFO|abc\n'
                          'log:WARNING|abc\n'
                          'log:ERROR|abc\n'
                          'log:CRITICAL|abc')

    def test_log_on(self):
        self.lm.log_on = False
        self.lm.log_debug('abc')
        self.lm.log_info('abc')
        self.lm.log_warning('abc')
        self.lm.log_error('abc')
        self.lm.log_critical('abc')
        self.assertEquals(sys.stdout.getvalue().strip(), '')

if __name__ == '__main__':
    main(buffer=True)
