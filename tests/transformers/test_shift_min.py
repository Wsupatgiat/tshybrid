import pytest
import numpy as np
import pandas as pd

from tshybrid.transformers.simple_transformers import ShiftMin

DESIRED_MIN = [1, 0, -1, 10, -10, 999999.99, -999999.99, -1.0, 1.0]

def get_expected_shifted_series(series, desired_min):
	expected_min_value = np.min(series)
	expected_shifted_series = series - expected_min_value + desired_min

	return expected_shifted_series, expected_min_value



'''
testing the ability to shift the min of a series
'''

@pytest.mark.parametrize('desired_min', DESIRED_MIN)
def test_shift_min_transform_series(synthetic_series, desired_min):
	expected_shifted_series, expected_min_value = get_expected_shifted_series(synthetic_series, desired_min)

	shift_min_transformer = ShiftMin(desired_min=desired_min)
	created_shifted_series = shift_min_transformer.fit_transform(synthetic_series)

	assert created_shifted_series.equals(expected_shifted_series)


@pytest.mark.parametrize(
	'desired_min',
	[1, 0, -1, 10, -10, 999999.99, -999999.99, -1.0, 1.0]
)
def test_shift_min_min_value_series(synthetic_series, desired_min):
	expected_shifted_series, expected_min_value = get_expected_shifted_series(synthetic_series, desired_min)

	shift_min_transformer = ShiftMin(desired_min=desired_min)
	created_shifted_series = shift_min_transformer.fit(synthetic_series)

	assert shift_min_transformer.min_value == expected_min_value

