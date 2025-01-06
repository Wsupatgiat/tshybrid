import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal


from tshybrid.transformers.simple_transformers import ShiftMin


def get_expected_shifted_series(series, desired_min):
	series = series.copy()
	expected_min_value = np.min(series)
	expected_shifted_series = series - expected_min_value + desired_min

	return expected_shifted_series, expected_min_value

def get_expected_shifted_df(df, desired_min, target_column):
	expected_shifted_df = df.copy()

	target_series = expected_shifted_df[target_column]
	expected_shifted_series, expected_min_value = get_expected_shifted_series(target_series, desired_min)
	expected_shifted_df[target_column] = expected_shifted_series

	return expected_shifted_df, expected_min_value

@pytest.fixture(params=[1, 0, -1, 10, -10, 999999.99, -999999.99, -1.0, 1.0])
def desired_min(request):
	return request.param

@pytest.fixture(params=['endog', 'mock_column'])
def target_column(request):
	return request.param


def test_shift_min_transform_series(synthetic_series, desired_min):
	expected_shifted_series, _ = get_expected_shifted_series(synthetic_series, desired_min)

	shift_min_transformer = ShiftMin(desired_min=desired_min)
	created_shifted_series = shift_min_transformer.fit_transform(synthetic_series)

	assert_series_equal(created_shifted_series, expected_shifted_series, check_exact=True)

def test_shift_min_minimum_series(synthetic_series, desired_min):
	'''
	test both original and new minimum
	'''
	_, expected_min_value = get_expected_shifted_series(synthetic_series, desired_min)

	shift_min_transformer = ShiftMin(desired_min=desired_min)
	created_shifted_series = shift_min_transformer.fit_transform(synthetic_series)

	created_shifted_series_min = np.min(created_shifted_series)

	assert shift_min_transformer.min_value == expected_min_value
	assert created_shifted_series_min == desired_min

def test_shift_min_inverse_transform_series(synthetic_series, desired_min):
	shift_min_transformer = ShiftMin(desired_min=desired_min)
	created_shifted_series = shift_min_transformer.fit_transform(synthetic_series)
	created_inversed_series = shift_min_transformer.inverse_transform(created_shifted_series)

	assert_series_equal(created_inversed_series, synthetic_series, check_exact=False)

def test_shift_min_alter_original_series(synthetic_series, desired_min):
	expected_original_series = synthetic_series.copy()
	shift_min_transformer = ShiftMin(desired_min=desired_min)

	created_shifted_series = shift_min_transformer.fit_transform(synthetic_series)
	assert_series_equal(synthetic_series, expected_original_series, check_exact=True)

	created_inversed_series = shift_min_transformer.inverse_transform(created_shifted_series)
	assert_series_equal(synthetic_series, expected_original_series, check_exact=True)


def test_shift_min_transform_dataframe(synthetic_dataframe, desired_min, target_column):
	expected_shifted_df, _ = get_expected_shifted_df(synthetic_dataframe, desired_min, target_column)

	shift_min_transformer = ShiftMin(desired_min=desired_min, target_column=target_column)
	created_shifted_df = shift_min_transformer.fit_transform(synthetic_dataframe)
	
	assert_frame_equal(created_shifted_df, expected_shifted_df, check_exact=True)


def test_shift_min_minimum_dataframe(synthetic_dataframe, desired_min, target_column):
	_, expected_min_value = get_expected_shifted_df(synthetic_dataframe, desired_min, target_column)

	shift_min_transformer = ShiftMin(desired_min=desired_min, target_column=target_column)
	created_shifted_df = shift_min_transformer.fit_transform(synthetic_dataframe)

	created_shifted_df_min = np.min(created_shifted_df[target_column])

	assert shift_min_transformer.min_value == expected_min_value
	assert created_shifted_df_min == desired_min

def test_shift_min_inverse_transform_dataframe(synthetic_dataframe, desired_min, target_column):
	shift_min_transformer = ShiftMin(desired_min=desired_min, target_column=target_column)
	created_shifted_df = shift_min_transformer.fit_transform(synthetic_dataframe)
	created_inversed_df = shift_min_transformer.inverse_transform(created_shifted_df)

	assert_frame_equal(created_inversed_df, synthetic_dataframe, check_exact=False)

def test_shift_min_alter_original_dataframe(synthetic_dataframe, desired_min):
	target_column = 'mock_column'
	expected_original_dataframe = synthetic_dataframe.copy()
	shift_min_transformer = ShiftMin(desired_min=desired_min, target_column=target_column)

	created_shifted_df = shift_min_transformer.fit_transform(synthetic_dataframe)
	assert_frame_equal(synthetic_dataframe, expected_original_dataframe, check_exact=True)

	created_inversed_df = shift_min_transformer.inverse_transform(created_shifted_df)
	assert_frame_equal(synthetic_dataframe, expected_original_dataframe, check_exact=True)

