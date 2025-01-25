import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import STL

from tshybrid.datasets.load_data import generate_synthetic_values
from tshybrid.transformers.seasonal_decompositors import STLDecompose

np.random.seed(42)

data = generate_synthetic_values(
	'MS', 48, '2021-01-01',
	trend_slope=20,
	base_value=-100,
	variance=50,
	seasonality_amplitude=100,
	seasonal_period=12,
	column_name='endog'
)

data_series = generate_synthetic_values(
	'MS', 48, '2021-01-01',
	trend_slope=20,
	base_value=-100,
	variance=50,
	seasonality_amplitude=100,
	seasonal_period=12
)


data['mock'] = 0

new = data.drop(columns=['mock'])
print(type(new))

# print(data)

# transformer = STLDecompose(period=12)
# cre = transformer.fit_transform(data_series)

# merged = pd.merge(cre, data, left_index=True, right_index=True)
# print(merged)


# init_params = {'period': 12}
# # fit_params = {'inner_iter': 30, 'outer_iter': 10}
# fit_params = {}

# stl = STL(data, **init_params)
# res = stl.fit()
# fig = res.plot()

# trend = res.trend
# seasonal = res.seasonal
# resid = res.resid

# test_op = pd.DataFrame({
# 	'trend': res.trend,
# 	'season': res.seasonal,
# 	'residual': res.resid
# })



plt.savefig('plot.png')

# decomp = transformer.fit_transform(data)

