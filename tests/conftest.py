import pytest
import numpy as np
import pandas as pd

@pytest.fixture(autouse=True)
def set_random_seed():
	np.random.seed(42)

@pytest.fixture()
def monthly_seasonal_synthetic_series():
	trend_slope = 20
	base_value = 100
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

	prediction_date_range = pd.date_range(start='2025-01-01', end='2026-06-01')
	prediction_series = pd.Series([0]*len(prediction_date_range), index=prediction_date_range)

	return synthetic_series, prediction_series







