import pytest
import numpy as np

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