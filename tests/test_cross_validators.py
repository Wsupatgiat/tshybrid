import pytest
import pandas as pd
from tshybrid.cross_validators import SlidingWindowCV


#TODO add test_case errpr
@pytest.mark.parametrize(
	('train_size', 'horizon'),
	[
		(10, 1),
		(10, 5),
		(1, 10),
		(5, 10),
		(1, 1)
	]
)
def test_sliding_window_cv(synthetic_series, train_size, horizon):

	first_train_value_list = []
	last_train_value_list = []

	first_test_value_list = []
	last_test_value_list = []

	cv = SlidingWindowCV(train_size=train_size, horizon=horizon)
	
	for fold_index, (train_index, test_index) in enumerate(cv.split(synthetic_series)):
		train, test = synthetic_series.iloc[train_index], synthetic_series.iloc[test_index]

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


	expected_first_train_values = synthetic_series.iloc[:-train_size-horizon+1]
	expected_last_train_values = synthetic_series.iloc[train_size-1:-horizon]

	if -horizon+1 != 0:
		expected_first_test_values = synthetic_series.iloc[train_size:-horizon+1]
	else:
		expected_first_test_values = synthetic_series.iloc[train_size:]


	expected_last_test_values = synthetic_series.iloc[train_size+horizon-1:]

	assert created_first_train_values.equals(expected_first_train_values)
	assert created_last_train_values.equals(expected_last_train_values)

	assert created_first_test_values.equals(expected_first_test_values)
	assert created_last_test_values.equals(expected_last_test_values)

	assert fold_index + 1 == cv.get_n_splits(synthetic_series)




