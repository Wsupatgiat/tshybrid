import pytest
import numpy as np
import pandas as pd

@pytest.fixture(autouse=True)
def set_random_seed():
	np.random.seed(42)

@pytest.fixture()
def synthetic_series():
	trend_slope = 20
	base_value = -100
	variance = 100
	seasonality_amplitude = 100
	seasonal_period = 12

	date_range = pd.date_range(start='2021-01-01', end='2024-12-01', freq='MS')
	length = len(date_range)

	trend = base_value + trend_slope*np.arange(length)
	seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(length) / seasonal_period)
	residuals = np.random.normal(0, variance, size=length)

	synthetic_data = trend + seasonality + residuals
	synthetic_series = pd.Series(synthetic_data, index=date_range)

	return synthetic_series

@pytest.fixture()
def prediction_series():
	trend_slope = 30
	base_value = -100 + (48*20)
	variance = 50
	seasonality_amplitude = 100
	seasonal_period = 12

	date_range = pd.date_range(start='2025-01-01', end='2026-06-01', freq='MS')
	length = len(date_range)

	trend = base_value + trend_slope*np.arange(length)
	seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(length) / seasonal_period)
	residuals = np.random.normal(0, variance, size=length)

	synthetic_data = trend + seasonality + residuals
	synthetic_series = pd.Series(synthetic_data, index=date_range)

	return synthetic_series

@pytest.fixture()
def negative_synthetic_series():
	trend_slope = -20
	base_value = -1000
	variance = 50
	seasonality_amplitude = 100
	seasonal_period = 12

	date_range = pd.date_range(start='2021-01-01', end='2024-12-01', freq='MS')
	length = len(date_range)

	trend = base_value + trend_slope*np.arange(length)
	seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(length) / seasonal_period)
	residuals = np.random.normal(0, variance, size=length)

	synthetic_data = trend + seasonality + residuals
	synthetic_series = pd.Series(synthetic_data, index=date_range)

	return synthetic_series

@pytest.fixture()
def positive_synthetic_series():
	trend_slope = 20
	base_value = 1000
	variance = 50
	seasonality_amplitude = 100
	seasonal_period = 12

	date_range = pd.date_range(start='2021-01-01', end='2024-12-01', freq='MS')
	length = len(date_range)

	trend = base_value + trend_slope*np.arange(length)
	seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(length) / seasonal_period)
	residuals = np.random.normal(0, variance, size=length)

	synthetic_data = trend + seasonality + residuals
	synthetic_series = pd.Series(synthetic_data, index=date_range)

	return synthetic_series



@pytest.fixture()
def synthetic_dataframe():
	trend_slope = 20
	base_value = -100
	variance = 100
	seasonality_amplitude = 100
	seasonal_period = 12

	date_range = pd.date_range(start='2021-01-01', end='2024-12-01', freq='MS')
	length = len(date_range)

	trend = base_value + trend_slope*np.arange(length)
	seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(length) / seasonal_period)
	residuals = np.random.normal(0, variance, size=length)

	synthetic_data = trend + seasonality + residuals
	synthetic_dataframe = pd.DataFrame({'endog': synthetic_data}, index=date_range)
	synthetic_dataframe['mock_column'] = synthetic_dataframe * 0.5


	return synthetic_dataframe

@pytest.fixture()
def prediction_dataframe():
	trend_slope = 30
	base_value = -100 + (48*20)
	variance = 50
	seasonality_amplitude = 100
	seasonal_period = 12

	date_range = pd.date_range(start='2025-01-01', end='2026-06-01', freq='MS')
	length = len(date_range)

	trend = base_value + trend_slope*np.arange(length)
	seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(length) / seasonal_period)
	residuals = np.random.normal(0, variance, size=length)

	synthetic_data = trend + seasonality + residuals
	synthetic_dataframe = pd.DataFrame({'endog': synthetic_data}, index=date_range)
	synthetic_dataframe['mock_column'] = synthetic_dataframe * 0.5


	return synthetic_dataframe

@pytest.fixture()
def negative_monthly_seasonal_synthetic_dataframe():
	trend_slope = -20
	base_value = -1000
	variance = 50
	seasonality_amplitude = 100
	seasonal_period = 12

	date_range = pd.date_range(start='2021-01-01', end='2024-12-01', freq='MS')
	length = len(date_range)

	trend = base_value + trend_slope*np.arange(length)
	seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(length) / seasonal_period)
	residuals = np.random.normal(0, variance, size=length)

	synthetic_data = trend + seasonality + residuals
	synthetic_dataframe = pd.DataFrame({'endog': synthetic_data}, index=date_range)
	synthetic_dataframe['mock_column'] = synthetic_dataframe * 0.5


	return synthetic_dataframe

@pytest.fixture()
def positive_monthly_seasonal_synthetic_dataframe():
	trend_slope = 20
	base_value = 1000
	variance = 50
	seasonality_amplitude = 100
	seasonal_period = 12

	date_range = pd.date_range(start='2021-01-01', end='2024-12-01', freq='MS')
	length = len(date_range)

	trend = base_value + trend_slope*np.arange(length)
	seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(length) / seasonal_period)
	residuals = np.random.normal(0, variance, size=length)

	synthetic_data = trend + seasonality + residuals
	synthetic_dataframe = pd.DataFrame({'endog': synthetic_data}, index=date_range)
	synthetic_dataframe['mock_column'] = synthetic_dataframe * 0.5


	return synthetic_dataframe

