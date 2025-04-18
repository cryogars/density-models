
import pytest
import datetime
import numpy as np
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

def test_date_to_doy_default(converter):
    # Test October 1 (start of water year)
    date = datetime.datetime(2023, 10, 1)
    assert converter.date_to_DOY(date, origin=10, algorithm="default") == 1
    
    # Test October 2
    date = datetime.datetime(2023, 10, 2)
    assert converter.date_to_DOY(date, origin=10, algorithm="default") == 2
    
    # Test last day of September (end of water year)
    date = datetime.datetime(2024, 9, 30)
    assert converter.date_to_DOY(date, origin=10, algorithm="default") == 366  # Leap year 2024


def test_date_to_doy_sturm(converter):
    # Test October 1 (start of Sturm water year)
    date = datetime.datetime(2023, 10, 1)
    assert converter.date_to_DOY(date, origin=10, algorithm="Sturm") == -92
    
    # Test January 1
    date = datetime.datetime(2024, 1, 1)
    assert converter.date_to_DOY(date, origin=10, algorithm="Sturm") == 1
    
    # Test July 1 (outside Sturm range)
    date = datetime.datetime(2024, 7, 1)
    assert np.isnan(converter.date_to_DOY(date, origin=10, algorithm="Sturm"))


def test_invalid_date_type(converter):
    # Test invalid date type
    with pytest.raises(TypeError):
        converter.date_to_DOY("not-a-date", algorithm="default")

def test_invalid_algorithm(converter):
    # Test invalid algorithm
    date = datetime.datetime(2023, 10, 1)
    with pytest.raises(ValueError):
        converter.date_to_DOY(date, algorithm="invalid_algorithm")