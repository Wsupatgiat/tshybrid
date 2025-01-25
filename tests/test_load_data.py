import pytest
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal

from tshybrid.datasets.load_data import generate_synthetic_values



def test_monthly_synthetic_series_values(synthetic_series):
	np.random.seed(42)
	created_series = generate_synthetic_values(
		'MS', 48, '2021-01-01',
		trend_slope=20,
		base_value=-100,
		variance=100,
		seasonality_amplitude=100,
		seasonal_period=12
	)

	assert_series_equal(synthetic_series, created_series, check_exact=True)

def test_monthly_synthetic_dataframe_values(synthetic_dataframe):
	np.random.seed(42)

	created_dataframe = generate_synthetic_values(
		'MS', 48, '2021-01-01',
		trend_slope=20,
		base_value=-100,
		variance=100,
		seasonality_amplitude=100,
		seasonal_period=12,
		column_name='endog'
	)

	created_dataframe['mock_column'] = created_dataframe['endog'] * 0.5



	assert_frame_equal(synthetic_dataframe, created_dataframe, check_exact=True)
