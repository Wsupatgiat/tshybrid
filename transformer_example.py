import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from tshybrid.datasets.load_data import generate_synthetic_values
from tshybrid.transformers.simple_transformers import ShiftMin, RootTransformer

np.random.seed(42)
data = generate_synthetic_values(
	'MS', 10, '2021-01-01',
	trend_slope=20,
	base_value=1000,
	variance=50,
	seasonality_amplitude=100,
	seasonal_period=12,
	column_name='endog'
)

data['mock'] = data['endog'] * 1.5

data_cp = data.copy()


transformer = ShiftMin()
transformed_df = transformer.fit_transform(data)

print(data == data_cp)



