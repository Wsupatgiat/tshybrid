import pytest
import numpy as np
import pandas as pd

from tshybrid.transformers.simple_transformers import ShiftMin, RootTransformer

#TODO make parameters better or smth
@pytest.mark.parametrize(
	'desired_min',
	[1, 0, -1, 10, -10, 999999.99, -999999.99, -1.0, 1.0]
)
#TODO change test name to be more clear
def test_shift_min_using_series(monthly_seasonal_synthetic_series, desired_min):
	synthetic_series, _ = monthly_seasonal_synthetic_series

	min_value = np.min(synthetic_series)
	expected_shifted_series = synthetic_series - min_value + desired_min

	transformer = ShiftMin(desired_min=desired_min)
	created_shifted_series = transformer.fit_transform(synthetic_series)

	created_shift_amount = created_shifted_series - synthetic_series
	expected_shift_amount = pd.Series([desired_min - min_value]*len(synthetic_series), index=synthetic_series.index)

	created_inversed_series = transformer.inverse_transform(created_shifted_series)

	np.testing.assert_allclose(created_shift_amount, expected_shift_amount)
	np.testing.assert_allclose(created_inversed_series, synthetic_series)
	assert created_shifted_series.equals(expected_shifted_series)
	assert transformer.min_value == min_value

@pytest.mark.parametrize(
	'desired_min',
	[1, 0, -1, 10, -10, 999999.99, -999999.99, -1.0, 1.0]
)
def test_shift_min_using_dataframe(monthly_seasonal_synthetic_dataframe, desired_min):
	synthetic_dataframe, _ = monthly_seasonal_synthetic_dataframe
	synthetic_series = synthetic_dataframe['endog']

	min_value = np.min(synthetic_series)
	expected_shifted_series = synthetic_series - min_value + desired_min
	expected_shifted_dataframe = pd.DataFrame({'endog': expected_shifted_series})

	transformer = ShiftMin(desired_min=desired_min, target_column='endog')
	created_shifted_dataframe = transformer.fit_transform(synthetic_dataframe)

	created_shift_amount = created_shifted_dataframe['endog'] - synthetic_series
	expected_shift_amount = pd.Series([desired_min - min_value]*len(synthetic_dataframe), index=synthetic_dataframe.index)

	created_inversed_dataframe = transformer.inverse_transform(created_shifted_dataframe)

	np.testing.assert_allclose(created_shift_amount, expected_shift_amount)
	np.testing.assert_allclose(created_inversed_dataframe, synthetic_dataframe)
	assert created_shifted_dataframe.equals(expected_shifted_dataframe)
	assert transformer.min_value == min_value

def test_shift_min_return_type(monthly_seasonal_synthetic_series, monthly_seasonal_synthetic_dataframe):
	synthetic_dataframe, _ = monthly_seasonal_synthetic_dataframe
	synthetic_series, _ = monthly_seasonal_synthetic_series

	#TODO change var name to be more explicit for previous 2 tests
	shift_min_transformer = ShiftMin()
	# shift_min_transformer_series = ShiftMin()

	created_transformed_dataframe = shift_min_transformer.fit_transform(synthetic_dataframe)
	created_transformed_series = shift_min_transformer.fit_transform(synthetic_series)

	created_inversed_dataframe = shift_min_transformer.inverse_transform(created_transformed_dataframe)
	created_inversed_series = shift_min_transformer.inverse_transform(created_transformed_series)

	assert isinstance(created_transformed_dataframe, pd.DataFrame)
	assert isinstance(created_transformed_series, pd.Series)

	assert isinstance(created_inversed_dataframe, pd.DataFrame)
	assert isinstance(created_inversed_series, pd.Series)

# def test_shift_min_target_column(monthly_seasonal_synthetic_dataframe):
# 	synthetic_dataframe, _ = monthly_seasonal_synthetic_dataframe

# 	target_column = 'new_column'
# 	synthetic_dataframe[target_column] = synthetic_dataframe['endog']

# 	shift_min_transformer = ShiftMin(target_column=column_name)



	







@pytest.mark.parametrize(
	'degree',
	[2, 3, 4, 10]
)
def test_root_transformer_on_positive_values_using_series(positive_monthly_seasonal_synthetic_series, degree):
	synthetic_series, _ = positive_monthly_seasonal_synthetic_series

	expected_transformed_series = synthetic_series**(1/degree)

	root_transformer = RootTransformer(degree=degree)
	created_transformed_series = root_transformer.fit_transform(synthetic_series)

	assert created_transformed_series.equals(expected_transformed_series)


@pytest.mark.parametrize(
	'degree',
	[2, 3, 4, 10]
)
def test_root_transformer_inverse_transform_using_series(positive_monthly_seasonal_synthetic_series, degree):
	synthetic_series, _ = positive_monthly_seasonal_synthetic_series

	root_transformer = RootTransformer(degree=degree)
	created_transformed_series = root_transformer.fit_transform(synthetic_series)
	created_inversed_series = root_transformer.inverse_transform(created_transformed_series)

	np.testing.assert_allclose(created_inversed_series, synthetic_series)


@pytest.mark.parametrize(
	'degree',
	[-1, 0, 1]
)
def test_root_transformer_degree_error_using_series(positive_monthly_seasonal_synthetic_series, degree):
	synthetic_series, _ = positive_monthly_seasonal_synthetic_series

	root_transformer = RootTransformer(degree=degree)

	with pytest.raises(ValueError):
		root_transformer.fit(synthetic_series)

@pytest.mark.parametrize(
	'degree',
	[2, 4, 10]
)
def test_root_transformer_negative_error_using_series(monthly_seasonal_synthetic_series, degree):
	synthetic_series, _ = monthly_seasonal_synthetic_series

	root_transformer = RootTransformer(degree=degree)

	with pytest.raises(ValueError):
		root_transformer.fit(synthetic_series)



@pytest.mark.parametrize(
	'degree',
	[2, 3, 4, 10]
)
def test_root_transformer_on_positive_values_using_dataframe(positive_monthly_seasonal_synthetic_dataframe, degree):
	synthetic_dataframe, _ = positive_monthly_seasonal_synthetic_dataframe

	expected_transformed_dataframe = synthetic_dataframe**(1/degree)

	root_transformer = RootTransformer(degree=degree)
	created_transformed_dataframe = root_transformer.fit_transform(synthetic_dataframe)

	assert created_transformed_dataframe.equals(expected_transformed_dataframe)


@pytest.mark.parametrize(
	'degree',
	[2, 3, 4, 10]
)
def test_root_transformer_inverse_transform_using_dataframe(positive_monthly_seasonal_synthetic_dataframe, degree):
	synthetic_dataframe, _ = positive_monthly_seasonal_synthetic_dataframe

	root_transformer = RootTransformer(degree=degree)
	created_transformed_dataframe = root_transformer.fit_transform(synthetic_dataframe)
	created_inversed_dataframe = root_transformer.inverse_transform(created_transformed_dataframe)

	np.testing.assert_allclose(created_inversed_dataframe.values, synthetic_dataframe.values)


@pytest.mark.parametrize(
	'degree',
	[2, 4, 10]
)
def test_root_transformer_negative_error_using_dataframe(monthly_seasonal_synthetic_dataframe, degree):
	synthetic_dataframe, _ = monthly_seasonal_synthetic_dataframe

	root_transformer = RootTransformer(degree=degree)

	with pytest.raises(ValueError):
		root_transformer.fit(synthetic_dataframe)

def test_root_transformer_return_type(positive_monthly_seasonal_synthetic_dataframe, positive_monthly_seasonal_synthetic_series):
	synthetic_dataframe, _ = positive_monthly_seasonal_synthetic_dataframe
	synthetic_series, _ = positive_monthly_seasonal_synthetic_series

	root_transformer = RootTransformer()

	created_transformed_dataframe = root_transformer.fit_transform(synthetic_dataframe)
	created_transformed_series = root_transformer.fit_transform(synthetic_series)

	created_inversed_dataframe = root_transformer.inverse_transform(created_transformed_dataframe)
	created_inversed_series = root_transformer.inverse_transform(created_transformed_series)

	assert isinstance(created_transformed_dataframe, pd.DataFrame)
	assert isinstance(created_transformed_series, pd.Series)

	assert isinstance(created_inversed_dataframe, pd.DataFrame)
	assert isinstance(created_inversed_series, pd.Series)




