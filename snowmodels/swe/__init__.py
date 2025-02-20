"""
The :mod:`snowmodels.swe` module includes statistical and machine learning models for computing snow water equivalent.
"""

from ._statistical_models import HillSWE, SWE_Models

__all__ = [
    'HillSWE',
    'SWE_Models'
]