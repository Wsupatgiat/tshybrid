import pytest
from tshybrid.datasets.load_data import generate_synthetic_series
import numpy as np



def test_monthly_synthetic_data(monthly_seasonal_synthetic_series):
	np.random.seed(42)
	synthetic_series, _ = monthly_seasonal_synthetic_series
	generated_series = generate_synthetic_series(
		'MS', 48, '2021-01-01',
		trend_slope=20,
		base_value=100,
		variance=100,
		seasonality_amplitude=100,
		seasonal_period=12
	)

	assert synthetic_series.equals(generated_series)

