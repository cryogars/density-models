
import pytest
from snowmodels.utils._jonas_model_constants import jonas_model_params, validate_month, MONTH_MAPPING


def test_jonas_model_params_structure():
    # Test the structure of the jonas_model_params dictionary
    assert len(jonas_model_params) == 12  # Should have all 12 months
    
    # Check each month has 3 elevation categories
    for month, elevation_data in jonas_model_params.items():
        assert set(elevation_data.keys()) == {'>=2000m', '[1400, 2000)m', '<1400m'}
        
        # Each elevation should have 'a' and 'b' parameters
        for elevation, params in elevation_data.items():
            assert 'a' in params
            assert 'b' in params


def test_jonas_model_specific_values():
    # Test a few specific values to ensure they match the expected parameters
    assert jonas_model_params['january']['>=2000m'] == {"b": 206, "a": 52}
    assert jonas_model_params['march']['<1400m'] == {"b": 333, "a": 3}
    assert jonas_model_params['june']['>=2000m'] == {"b": 452, "a": 8}
    
    # Check that some values are None as expected
    assert jonas_model_params['august']['>=2000m'] == {"b": None, "a": None}
    assert jonas_model_params['october']['[1400, 2000)m'] == {"b": None, "a": None}


def test_validate_month_numeric():
    # Test numeric inputs
    assert validate_month('1') == 'january'
    assert validate_month('12') == 'december'
    assert validate_month('deC') == 'december'
    assert validate_month(12) == 'december'
    assert validate_month(3) == 'march'
    
    # Test invalid numeric inputs
    with pytest.raises(ValueError):
        validate_month('13')
    with pytest.raises(ValueError):
        validate_month(0)
    with pytest.raises(ValueError):
        validate_month("abc")