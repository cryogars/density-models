import pytest
import datetime
import numpy as np
import pandas as pd
from snowmodels.utils._conversions import OutOfBoundsError
from snowmodels.utils._sturm_model_constants import (
    sturm_model_params,
    validate_SturmDOY,
    VALID_SNOW_CLASSES,
    validate_snow_class
)

def test_sturm_model_params_structure():
    # Test the structure of the sturm_model_params dictionary
    assert len(sturm_model_params) == 5  # Should have all 5 snow classes
    
    # Check each snow class has the expected parameters
    for snow_class, params in sturm_model_params.items():
        assert set(params.keys()) == {'rho_max', 'rho_0', 'k1', 'k2'}


def test_sturm_model_parameters():
    # Test specific values from each snow class
    assert sturm_model_params['alpine']['rho_max'] == 0.5975
    assert sturm_model_params['alpine']['rho_0'] == 0.2237
    assert sturm_model_params['alpine']['k1'] == 0.0012
    assert sturm_model_params['alpine']['k2'] == 0.0038
    assert sturm_model_params['maritime']['rho_0'] == 0.2578
    assert sturm_model_params['prairie']['k1'] == 0.0016
    assert sturm_model_params['tundra']['k2'] == 0.0049
    assert sturm_model_params['taiga']['rho_max'] == 0.2170
    assert sturm_model_params['taiga']['k1'] == 0.0000

def test_validate_snow_class():
    # Test valid snow classes
    for snow_class in VALID_SNOW_CLASSES:
        assert validate_snow_class(snow_class) == snow_class.lower()
    
    # Test case insensitivity
    assert validate_snow_class('ALPINE') == 'alpine'
    
    # Test invalid snow class
    assert np.isnan(validate_snow_class('invalid_class'))


def test_validate_sturm_doy_integers():
    # Test valid integer DOYs
    assert validate_SturmDOY(-50) == -50
    assert validate_SturmDOY(0) == 0
    assert validate_SturmDOY(150) == 150
    
    # Test integer-like strings
    assert validate_SturmDOY("100") == 100
    
    # Test out of range DOYs
    with pytest.raises(OutOfBoundsError):
        validate_SturmDOY(-100)
    
    with pytest.raises(OutOfBoundsError):
        validate_SturmDOY(200)
    
    # Test non-integer DOYs
    with pytest.raises(ValueError):
        validate_SturmDOY(10.5)


def test_validate_sturm_doy_datetime():
    # Test October 1st (should be -92)
    oct_1 = datetime.datetime(2023, 10, 1)
    assert validate_SturmDOY(oct_1) == -92

    # Test same date in a different year (should give same result)
    oct_1_different_year = datetime.datetime(2022, 10, 1)
    assert validate_SturmDOY(oct_1_different_year) == -92
    
    # Test February 1st (should be 32)
    feb_1 = datetime.datetime(2024, 2, 1)
    assert validate_SturmDOY(feb_1) == 32

    # Test same date in a different year (should give same result)
    feb_1_different_year = datetime.datetime(2022, 2, 1)
    assert validate_SturmDOY(feb_1_different_year) == 32
    
    # Test November 15th (should be -47)
    nov_15 = datetime.datetime(1996, 11, 15)
    assert validate_SturmDOY(nov_15) == -47
    
    # Test using a string date
    assert validate_SturmDOY("2023-10-01") == -92
    
    # Test using a Pandas Timestamp
    assert validate_SturmDOY(pd.Timestamp("2024-02-01")) == 32
    
    # Test that July-September returns NaN (excluded months)
    july_1 = datetime.datetime(1987, 7, 1)
    assert np.isnan(validate_SturmDOY(july_1))
    
    
    
    