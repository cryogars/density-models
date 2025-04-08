
import pytest
import numpy as np
from snowmodels.density import JonasDensity
from snowmodels.utils._jonas_model_constants import jonas_model_params, validate_month, MONTH_MAPPING

@pytest.fixture
def jonas_model():
    return JonasDensity()

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


def test_validate_month():
    # Test numeric inputs
    assert validate_month('1') == 'january'
    assert validate_month('12') == 'december'
    assert validate_month(12) == 'december'
    assert validate_month(3) == 'march'

    # Test short month names
    assert validate_month('deC') == 'december'
    assert validate_month('JaN') == 'january'
    
    # Test with extra whitespace
    assert validate_month(' may ') == 'may'
    
    # Test invalid month name
    with pytest.raises(ValueError):
        validate_month('janu')

    # Test invalid numeric inputs
    with pytest.raises(ValueError):
        validate_month('13')
    with pytest.raises(ValueError):
        validate_month(0)
    with pytest.raises(ValueError):
        validate_month("abc")


def test_month_mapping_completeness():
    # Ensure MONTH_MAPPING has all the expected keys
    expected_keys = set([str(i) for i in range(1, 13)] + 
                       ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    assert set(MONTH_MAPPING.keys()) == expected_keys
    
    # Ensure all values are valid full month names
    expected_values = set(['january', 'february', 'march', 'april', 'may', 'june', 
                           'july', 'august', 'september', 'october', 'november', 'december'])
    assert set(MONTH_MAPPING.values()) == expected_values

def test_jonas_model_computation(jonas_model):
    # Test January at high elevation
    # a=52, b=206, snow_depth=1.0
    # density = (52 * 1.0 + 206)/1000 = 0.258
    density = jonas_model.compute_density(
        snow_depth=1.0,
        month="january",
        elevation=2500
    )
    assert round(density, 3) == 0.258
    
    # Test with different month (March) and mid-elevation
    # a=31, b=281, snow_depth=0.5
    # density = (31 * 0.51 + 281)/1000 = 0.29681
    density = jonas_model.compute_density(
        snow_depth=0.51,
        month="march",
        elevation=1600
    )
    assert round(density, 3) == 0.297
    
    # Test with numeric month
    # Same as January test but with month as "1"
    density = jonas_model.compute_density(
        snow_depth=1.0,
        month="1",
        elevation=2500
    )
    assert round(density, 3) == 0.258
    
    # Test with a month/elevation that has None parameters (e.g., June at low elevation)
    density = jonas_model.compute_density(
        snow_depth=1.0,
        month="june",
        elevation=1200
    )
    assert np.isnan(density)

def test_jonas_elevation_boundaries(jonas_model):
    # Test at the exact elevation boundaries
    
    # Just below 1400m should use <1400m parameters
    density_low = jonas_model.compute_density(
        snow_depth=1.0,
        month="january",
        elevation=1399.9
    )
    
    # Exactly at 1400m should use [1400, 2000)m parameters
    density_mid = jonas_model.compute_density(
        snow_depth=1.0,
        month="january",
        elevation=1400
    )
    
    # Just below 2000m should use [1400, 2000)m parameters
    density_mid2 = jonas_model.compute_density(
        snow_depth=1.0,
        month="january",
        elevation=1999.9
    )
    
    # Exactly at 2000m should use >=2000m parameters
    density_high = jonas_model.compute_density(
        snow_depth=1.0,
        month="january",
        elevation=2000
    )
    
    # Verify these are different (boundary cases work correctly)
    assert density_low != density_mid
    assert density_mid == density_mid2
    assert density_mid != density_high
