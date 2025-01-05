import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from tshybrid.datasets.load_data import generate_synthetic_values
from tshybrid.transformers.simple_transformers import ShiftMin, RootTransformer

np.random.seed(42)
data = generate_synthetic_values(
	'MS', 48, '2021-01-01',
	trend_slope=20,
	base_value=1000,
	variance=50,
	seasonality_amplitude=100,
	seasonal_period=12,
	column_name='asds'
)

print(data)
data['mock'] = 0
print(data)

# print(data > 0)

# data_min = np.min(data)

# shifted_data = data - data_min + 1


# # plt.plot(shifted_data, label='shifted')
# plt.plot(data, label='orig')



# '''
# check
# - range between shifted and orig
# - is the min the same
# - inverse transform
# '''
# print('----')
# print((shifted_data - data).eq(1 - data.min()))
# # print((shifted_data - data) == (1 - data.min()))

# diff = shifted_data - data
# print(diff.unique())
# print(1 - data.min())

# print('+++++')





# transformer = ShiftMin()
# transformed = transformer.fit_transform(data)

# root_transformer = RootTransformer()

# # print((data - transformed).unique())

# # print(transformer.min_value)

# # print(transformed.equals(shifted_data))

# plt.savefig('plot.png')
