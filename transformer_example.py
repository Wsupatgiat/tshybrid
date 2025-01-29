import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sklearn.pipeline import Pipeline

from tshybrid.datasets.load_data import generate_synthetic_values
from tshybrid.transformers.simple_transformers import ShiftMin, RootTransformer
from tshybrid.transformers.seasonal_decompositors import STLDecompose

from tshybrid.plots.plot_ts import plot_df_vert

np.random.seed(42)
data = generate_synthetic_values(
	'MS', 100, '2021-01-01',
	trend_slope=20,
	base_value=1000,
	variance=50,
	seasonality_amplitude=100,
	seasonal_period=12,
	column_name='endog'
)

'''
root transform --> STL decompose --> shift season min only
'''

pipeline = Pipeline([
	('root_transform', RootTransformer(degree=2)),
	('stl_decompose', STLDecompose(period=12)),
	('shift_season_min', ShiftMin(target_column='season'))
])

transformed = pipeline.fit_transform(data)

plot_df_vert(transformed)
plt.savefig('plot.png')
