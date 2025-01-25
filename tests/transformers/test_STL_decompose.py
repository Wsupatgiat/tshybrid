import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
from statsmodels.tsa.seasonal import STL

from tshybrid.transformers.seasonal_decompositors import STLDecompose


@pytest.fixture(params=[
	{'period': 12},
	{
		'period': 12,
		'seasonal': 5,
		'trend': 21,
		'low_pass': 13,
		'seasonal_deg': 3,
		'trend_deg': 2,
		'low_pass_deg': 2,
		'robust': True,
		'seasonal_jump': 2,
		'trend_jump': 2,
		'low_pass_jump': 2
	}
])
def stl_init_parameters(request):
	return request.param


@pytest.fixture(params=[
	{},
	{
		'inner_iter': 7,
		'outer_iter': 23
	}
])
def stl_fit_parameters(request):
	return request.param

#TODO move to con test
@pytest.fixture(params=['endog', 'mock_column'])
def target_column(request):
	return request.param

def get_expected_decomposed_series(series, stl_init_parameters, stl_fit_parameters):
	series = series.copy()
	stl = STL(series, **stl_init_parameters)
	stl_fit = stl.fit(**stl_fit_parameters)
	
	trend = stl_fit.trend
	season = stl_fit.seasonal
	residuals = stl_fit.resid
	
	expected_decomposed_series = pd.DataFrame({
		'trend': trend,
		'season': season,
		'residuals': residuals
	})
	
	return expected_decomposed_series

def get_expected_decomposed_df(df, stl_init_parameters, stl_fit_parameters, target_column):
	df = df.copy()

	target_series = df[target_column]
	expected_decomposed_series = get_expected_decomposed_series(
		target_series,
		stl_init_parameters,
		stl_fit_parameters
	)
	
	expected_decomposed_df = df.drop(columns=[target_column])
	expected_decomposed_df = pd.merge(
		expected_decomposed_df,
		expected_decomposed_series,
		left_index=True, right_index=True
	)
	
	return expected_decomposed_df


def test_STL_decompose_series(synthetic_series, stl_init_parameters, stl_fit_parameters):
	expected_decomposed_series = get_expected_decomposed_series(
		synthetic_series,
		stl_init_parameters,
		stl_fit_parameters
	)

	stl_decomposer = STLDecompose(**stl_init_parameters | stl_fit_parameters)
	created_decomposed_series = stl_decomposer.fit_transform(synthetic_series)

	assert_frame_equal(created_decomposed_series, expected_decomposed_series, check_exact=True)

def test_STL_decompose_dataframe(synthetic_dataframe, stl_init_parameters, stl_fit_parameters, target_column):
	expected_decomposed_df = get_expected_decomposed_df(
		synthetic_dataframe,
		stl_init_parameters,
		stl_fit_parameters,
		target_column
	)

	stl_decomposer = STLDecompose(target_column=target_column, **stl_init_parameters | stl_fit_parameters)
	created_decomposed_df = stl_decomposer.fit_transform(synthetic_dataframe)

	assert_frame_equal(created_decomposed_df, expected_decomposed_df, check_exact=True)

def test_STL_inverse_decompose_dataframe(synthetic_dataframe, stl_init_parameters, stl_fit_parameters, target_column):
	stl_decomposer = STLDecompose(target_column=target_column, **stl_init_parameters | stl_fit_parameters)
	created_decomposed_df = stl_decomposer.fit_transform(synthetic_dataframe)
	created_inversed_df = stl_decomposer.inverse_transform(created_decomposed_df)


	assert_frame_equal(created_inversed_df, synthetic_dataframe, check_exact=False, check_like=True)

def test_STL_decompose_alter_original_dataframe(synthetic_dataframe, stl_init_parameters, stl_fit_parameters):
	expected_original_df = synthetic_dataframe.copy()

	stl_decomposer = STLDecompose(**stl_init_parameters | stl_fit_parameters)

	created_decomposed_df = stl_decomposer.fit_transform(synthetic_dataframe)
	assert_frame_equal(synthetic_dataframe, expected_original_df, check_exact=True)

	created_inversed_df = stl_decomposer.inverse_transform(created_decomposed_df)
	assert_frame_equal(synthetic_dataframe, expected_original_df, check_exact=True)







