
import pytest
import datetime
import numpy as np
import pandas as pd
from snowmodels.utils import ConvertData


@pytest.fixture
def converter():
    return ConvertData()


def test_fah_to_cel():
    # Test freezing point
    assert ConvertData.fah_to_cel(32) == 0
    
    # Test boiling point
    assert abs(ConvertData.fah_to_cel(212) - 100) < 0.0001
    
    # Test negative temperature
    assert ConvertData.fah_to_cel(-40) == -40

def test_feet_to_m():
    # Test 1 foot
    assert ConvertData.feet_to_m(1) == 0.3048
    
    # Test 10 feet
    assert ConvertData.feet_to_m(10) == 3.048

def test_inches_to_metric():
    # Test conversion to meters
    assert ConvertData.inches_to_metric(10, 'meters') == 0.254
    
    # Test conversion to cm
    assert ConvertData.inches_to_metric(10, 'cm') == 25.4
    
    # Test conversion to mm
    assert ConvertData.inches_to_metric(10, 'mm') == 254.0
    
    # Test invalid unit
    with pytest.raises(ValueError):
        ConvertData.inches_to_metric(10, 'invalid_unit') 