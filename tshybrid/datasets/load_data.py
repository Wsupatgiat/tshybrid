import os

import pandas as pd

DIRNAME = "data"
MODULE = os.path.dirname(__file__)

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
