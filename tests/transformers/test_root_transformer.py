import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from tshybrid.transformers.simple_transformers import RootTransformer


'''
there is probably a better way to test all of this, will rewrite later
'''
@pytest.fixture(params=[2, 3, 4, 5, 10, 11])
def degree(request):
	return request.param

@pytest.fixture(params=[2, 4, 8, 10])
def degree_even(request):
	return request.param

@pytest.fixture(params=[-2, -1, 0, 1])
def degree_error_trigger(request):
	return request.param


def get_expected_root_transformed_series(series, degree):
	series = series.copy()
	expected_root_transformed_series = series**(1/degree)
	return expected_root_transformed_series

def get_expected_root_transformed_df(df, degree, target_column):
	expected_transformed_df = df.copy()

	target_series = expected_transformed_df[target_column]
	expected_transformed_series = get_expected_root_transformed_series(target_series, degree)

	expected_transformed_df[target_column] = expected_transformed_series
	return expected_transformed_df



def test_root_transform_series(positive_synthetic_series, degree):
	expected_transformed_series = get_expected_root_transformed_series(positive_synthetic_series, degree)

	transformer = RootTransformer(degree=degree)
	created_transformed_series = transformer.fit_transform(positive_synthetic_series)

	assert_series_equal(created_transformed_series, expected_transformed_series, check_exact=True)

def test_root_inverse_transform_series(positive_synthetic_series, degree):
	transformer = RootTransformer(degree=degree)
	created_transformed_series = transformer.fit_transform(positive_synthetic_series)
	created_inversed_series = transformer.inverse_transform(created_transformed_series)
	
	assert_series_equal(created_inversed_series, positive_synthetic_series, check_exact=False)

def test_root_transform_alter_original_series(positive_synthetic_series, degree):
	expected_original_series = positive_synthetic_series.copy()
	transformer = RootTransformer(degree=degree)

	created_transformed_series = transformer.fit_transform(positive_synthetic_series)
	assert_series_equal(positive_synthetic_series, expected_original_series, check_exact=True)

	created_inversed_series = transformer.inverse_transform(created_transformed_series)
	assert_series_equal(positive_synthetic_series, expected_original_series, check_exact=True)

def test_root_transform_degree_error_series(positive_synthetic_series, degree_error_trigger):
	transformer = RootTransformer(degree=degree_error_trigger)
	with pytest.raises(ValueError):
		transformer.fit(positive_synthetic_series)

def test_root_transform_even_degree_negative_series(synthetic_series, degree_even):
	transformer = RootTransformer(degree=degree_even)
	with pytest.raises(ValueError):
		transformer.fit(synthetic_series)

def test_root_transform_dataframe(positive_synthetic_dataframe, degree, target_column):
	expected_transformed_df = get_expected_root_transformed_df(positive_synthetic_dataframe, degree, target_column)

	transformer = RootTransformer(degree=degree, target_column=target_column)
	created_transformed_df = transformer.fit_transform(positive_synthetic_dataframe)

	assert_frame_equal(created_transformed_df, expected_transformed_df, check_exact=True)

def test_root_inverse_transform_dataframe(positive_synthetic_dataframe, degree, target_column):
	transformer = RootTransformer(degree=degree, target_column=target_column)
	created_transformed_df = transformer.fit_transform(positive_synthetic_dataframe)
	created_inversed_df = transformer.inverse_transform(created_transformed_df)

	assert_frame_equal(created_inversed_df, positive_synthetic_dataframe, check_exact=False)

def test_root_transform_alter_original_dataframe(positive_synthetic_dataframe, degree, target_column):
	expected_original_df = positive_synthetic_dataframe.copy()
	transformer = RootTransformer(degree=degree, target_column=target_column)

	created_transformed_df = transformer.fit_transform(positive_synthetic_dataframe)
	assert_frame_equal(positive_synthetic_dataframe, expected_original_df, check_exact=True)

	created_inverse_df = transformer.inverse_transform(created_transformed_df)
	assert_frame_equal(positive_synthetic_dataframe, expected_original_df, check_exact=True)

def test_root_transform_degree_error_dataframe(positive_synthetic_dataframe, degree_error_trigger, target_column):
	transformer = RootTransformer(degree=degree_error_trigger, target_column=target_column)
	with pytest.raises(ValueError):
		transformer.fit(positive_synthetic_dataframe)

def test_root_transform_even_degree_negative_dataframe(synthetic_dataframe, degree_even, target_column):
	transformer = RootTransformer(degree=degree_even, target_column=target_column)
	with pytest.raises(ValueError):
		transformer.fit(synthetic_dataframe)
















