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
    log_on : bool
        Initial value to assign to class instance's `log_on` property.

    Attributes
    ----------
    log_on : bool
        If set to False, the logger's methods will silently
        do nothing when called.

    Methods
    -------
    log_debug(), log_info(), log_warning(), log_error(), log_critical()
        Emit a log message at the level corresponding to the method name.
    """

    def __init__(self, name, log_on=True):
        super(LoggerMixin, self).__init__()
        self.logger = twiggy.log.name(name)
        self.log_on = log_on

    @property
    def log_on(self):
        """
        Logger switch. If False, the logging methods silently do nothing.
        """

        return self._log_on

    @log_on.setter
    def log_on(self, value):
        self._log_on = bool(value)
        if self._log_on:
            self.log_debug = self.logger.debug
            self.log_info = self.logger.info
            self.log_warning = self.logger.warning
            self.log_error = self.logger.error
            self.log_critical = self.logger.critical
        else:
            self.log_debug = lambda x: None
            self.log_info = lambda x: None
            self.log_warning = lambda x: None
            self.log_error = lambda x: None
            self.log_critical = lambda x: None

if __name__ == '__main__':
    import sys
    output = twiggy.outputs.StreamOutput(twiggy.formats.line_format,
                                     stream=sys.stdout)
    twiggy.emitters['*'] = twiggy.filters.Emitter(twiggy.levels.DEBUG, True, output)

    l = LoggerMixin('foo')
    l.log_info('test')
    l.log_on = False
    l.log_info('test')
