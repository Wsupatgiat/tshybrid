import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.seasonal import STL

from tshybrid.datasets.load_data import generate_synthetic_values

data = generate_synthetic_values(
	'MS', 48, '2021-01-01',
	trend_slope=20,
	base_value=-100,
	variance=0,
	seasonality_amplitude=100,
	seasonal_period=12
)

stl = STL(data, period=12)
res = stl.fit()
# fig = res.plot()

trend = res.trend
seasonal = res.seasonal
resid = res.resid

# plt.plot(trend)
plt.savefig('plot.png')
