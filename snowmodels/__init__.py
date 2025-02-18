"""
Top-level package for snowmodels.
"""

__author__ = """Ibrahim Alabi"""
__email__ = "ibrahimolalekana@u.boisestate.edu"
__version__ = "0.0.1"


_submodules = [
    'swe',
    'density'
]

__all__ = _submodules

def __dir__():
    return __all__