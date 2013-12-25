try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)

# Ignore all exceptions so that this doesn't cause package installation
# to fail if pkg_resources can't find neurokernel:
try:
    from version import __version__
except:
    pass
