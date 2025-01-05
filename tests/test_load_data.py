import pytest
from tshybrid.datasets.load_data import generate_synthetic_values
import numpy as np



def test_monthly_synthetic_series_values(monthly_seasonal_synthetic_series):
	np.random.seed(42)
	synthetic_series, _ = monthly_seasonal_synthetic_series
	created_series = generate_synthetic_values(
		'MS', 48, '2021-01-01',
		trend_slope=20,
		base_value=-100,
		variance=100,
		seasonality_amplitude=100,
		seasonal_period=12
	)

	assert synthetic_series.equals(created_series)

def test_monthly_synthetic_dataframe_values(monthly_seasonal_synthetic_dataframe):
	np.random.seed(42)
	synthetic_dataframe, _ = monthly_seasonal_synthetic_dataframe

	created_dataframe = generate_synthetic_values(
		'MS', 48, '2021-01-01',
		trend_slope=20,
		base_value=-100,
		variance=100,
		seasonality_amplitude=100,
		seasonal_period=12,
		column_name='endog'
	)

	assert synthetic_dataframe.equals(created_dataframe)
