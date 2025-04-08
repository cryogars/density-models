
# tests/test_pistocchi_model.py
import pytest
import datetime
import numpy as np
import pandas as pd
from snowmodels.density import PistochiDensity

@pytest.fixture
def pistocchi_model():
    return PistochiDensity()

def test_pistocchi_model_computation(pistocchi_model):
    # Test specific day values
    # Day 1 (November 1) should give a density of (200 + (1 + 61))/1000 = 0.262
    assert pistocchi_model.compute_density(DOY=1) == 0.262
    
    # Day 100 should give a density of (200 + (100 + 61))/1000 = 0.361
    assert pistocchi_model.compute_density(DOY=100) == 0.361
    
    # Test with date objects
    nov_15 = datetime.datetime(2023, 11, 15)
    # Should convert to day 15 and give density of (200 + (15 + 61))/1000 = 0.276
    assert round(pistocchi_model.compute_density(nov_15), 3) == 0.276
    
    # Test with string date
    # Should convert to day 31 and give density of (200 + (31 + 61))/1000 = 0.292
    assert round(pistocchi_model.compute_density("2023-12-01"), 3) == 0.292