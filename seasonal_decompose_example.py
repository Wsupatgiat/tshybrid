import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import STL
from sklearn.pipeline import Pipeline

from tshybrid.datasets.load_data import generate_synthetic_values
from tshybrid.transformers.simple_transformers import RootTransformer
from tshybrid.transformers.seasonal_decompositors import STLDecompose

np.random.seed(42)

data = generate_synthetic_values(
	'MS', 48, '2021-01-01',
	trend_slope=20,
	base_value=100,
	variance=10,
	seasonality_amplitude=100,
	seasonal_period=12,
	# column_name='endog'
)

pipeline = Pipeline([
	('RootTransform', RootTransformer(degree=2))
])

plt.plot(data)

transformed = pipeline.fit_transform(data)
print(transformed)


plt.savefig('plot.png')


