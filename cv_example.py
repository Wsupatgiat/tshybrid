import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tshybrid.datasets.load_data import generate_synthetic_series
from tshybrid.cross_validators.sliding_cross_validators import SlidingWindowCV

'''
generate synthetic data
'''
data = generate_synthetic_series(
	'MS', 48, '2021-01-01',
	trend_slope=20,
	base_value=-100,
	variance=100,
	seasonality_amplitude=100,
	seasonal_period=12
)


'''
create sliding cv object
'''

'''
train = 3
test = 2
0  1  2  3  4  5  6  7  8  9
[.....]  [..]

0  1  2  3  4  5  6  7  8  9
   [.....]  [..]

0  1  2  3  4  5  6  7  8  9
      [.....]  [..]

0  1  2  3  4  5  6  7  8  9
         [.....]  [..]

0  1  2  3  4  5  6  7  8  9
            [.....]  [..]

0  1  2  3  4  5  6  7  8  9
               [.....]  [..]
'''

train_size = 3
horizon = 1


cv = SlidingWindowCV(train_size=train_size, horizon=horizon)


first_train_value_list = []
last_train_value_list = []

first_test_value_list = []
last_test_value_list = []

print(data.index)
for fold_index, (train_index, test_index) in enumerate(cv.split(data)):
	train, test = data.iloc[train_index], data.iloc[test_index]

	print(f"train: {train.index[0]} - {train.index[-1]}")
	print(f"test: {test.index[0]} - {test.index[-1]}")

	first_train_value = train.iloc[[0]]
	last_train_value = train.iloc[[-1]]

	first_test_value = test.iloc[[0]]
	last_test_value = test.iloc[[-1]]

	first_train_value_list.append(first_train_value)
	last_train_value_list.append(last_train_value)

	first_test_value_list.append(first_test_value)
	last_test_value_list.append(last_test_value)

created_first_train_values = pd.concat(first_train_value_list)
created_last_train_values = pd.concat(last_train_value_list)

created_first_test_values = pd.concat(first_test_value_list)
created_last_test_values = pd.concat(last_test_value_list)


actual_first_train_values = data.iloc[:-train_size-horizon+1]
actual_last_train_values = data.iloc[train_size-1:-horizon]

if -horizon + 1 != 0:
	actual_first_test_values = data.iloc[train_size:-horizon+1]
else:
	actual_first_test_values = data.iloc[train_size:]


actual_last_test_values = data.iloc[train_size+horizon-1:]

print('----')
print(actual_first_test_values)
print(created_first_test_values)

print(created_first_train_values.equals(actual_first_train_values))
print(created_last_train_values.equals(actual_last_train_values))

print(created_first_test_values.equals(actual_first_test_values))
print(created_last_test_values.equals(actual_last_test_values))


print('---')







#Check n_splits
print(fold_index+1 == cv.get_n_splits(data))
print(cv.get_n_splits(data))








plt.plot(data)
plt.savefig('plot.png')
