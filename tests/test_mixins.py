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

    def test_debug(self):
        self.lm.debug('abc')
        self.lm.info('abc')
        self.lm.warning('abc')
        self.lm.error('abc')
        self.lm.critical('abc')

        # All output to stdout is buffered within a single test before emission:
        self.assertEquals(sys.stdout.getvalue().strip(), 
                          'log:DEBUG:abc\n'
                          'log:INFO:abc\n'
                          'log:WARNING:abc\n'
                          'log:ERROR:abc\n'
                          'log:CRITICAL:abc')

    def test_on(self):
        self.lm.on = False
        self.lm.debug('abc')
        self.lm.info('abc')
        self.lm.warning('abc')
        self.lm.error('abc')
        self.lm.critical('abc')
        self.assertEquals(sys.stdout.getvalue().strip(), '')
        
if __name__ == '__main__':
    main(buffer=True)
