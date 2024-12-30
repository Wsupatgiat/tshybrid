import os

import pandas as pd
import numpy as np

DIRNAME = "data"
MODULE = os.path.dirname(__file__)


def generate_synthetic_series(
		freq,
		length,
		start_date,
		trend_slope=0,
		base_value=0,
		variance=0,
		seasonality_amplitude=0, 
		seasonal_period=1,
	):

	date_range = pd.date_range(start=start_date, periods=length, freq=freq)

	trend = base_value + trend_slope*np.arange(length)
	seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(length) / seasonal_period)
	residuals = np.random.normal(0, variance, size=length)

	synthetic_data = trend + seasonality + residuals
	synthetic_series = pd.Series(synthetic_data, index=date_range)

	return synthetic_series

def load_air_passegers():
	name = "AirPassengers"
	fname = name + '.csv'
	path = os.path.join(MODULE, DIRNAME, fname)
	csv_data = pd.read_csv(path, index_col=0, dtype={1:float}).squeeze()

	csv_data.index = pd.to_datetime(csv_data.index)
	csv_data.name = name
	csv_data.index.name = 'date'
	csv_data = csv_data.asfreq('MS')

	return csv_data
