#!/usr/bin/env python

"""
Object with built-in logger.
"""

import twiggy

class LoggerMixin(object):
    """
    Mixin that provides a per-instance logger that can be turned off.

    Parameters
    ----------
    name : str
        Name to assign logger.
    on : bool
        Initial value to assign to class instance's `on` property.

    Attributes
    ----------
    on : bool
        If set to False, the logger's methods will silently 
        do nothing when called.
    """

    def __init__(self, name, on=True):
        super(LoggerMixin, self).__init__()
        self.logger = twiggy.log.name(name)
        self.on = on

    @property
    def on(self):
        """
        Logger switch. If False, the logging methods silently do nothing.
        """

        return self._on

    @on.setter
    def on(self, value):
        self._on = bool(value)
        if self._on:
            self.debug = self.logger.debug
            self.info = self.logger.info
            self.warning = self.logger.warning
            self.error = self.logger.error
            self.critical = self.logger.critical
        else:
            self.debug = lambda x: None
            self.info = lambda x: None
            self.warning = lambda x: None
            self.error = lambda x: None
            self.critical = lambda x: None

if __name__ == '__main__':
    import sys
    output = twiggy.outputs.StreamOutput(twiggy.formats.line_format,
                                     stream=sys.stdout)
    twiggy.emitters['*'] = twiggy.filters.Emitter(twiggy.levels.DEBUG, True, output)

    l = LoggerMixin('foo')
    l.info('test')
    l.on = False
    l.info('test')
