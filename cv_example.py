import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tshybrid.datasets.load_data import generate_synthetic_series
from tshybrid.cross_validators.sliding_cross_validators import SlidingWindowCV

'''
generate synthetic data
'''
data = generate_synthetic_series(
	'MS', 20, '2021-01-01',
	trend_slope=20,
	base_value=10,
	variance=10
)


'''
create sliding cv object
'''
cv = SlidingWindowCV(train_size=5, horizon=2)

for train_index, test_index in cv.split(data):
	train, test = data.iloc[train_index], data.iloc[test_index]
	print(train_index)
	print(test_index)








