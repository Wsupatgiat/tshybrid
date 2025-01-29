import pytest
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tshybrid.predictors.wrappers import StatsmodelsWrapper


def test_initialize_sm_wrapper_parameters_as_kwargs_using_series(synthetic_series, prediction_series):
	init_params = {
		'trend': 'add',
		'damped_trend': False,
		'seasonal': 'add',
		'seasonal_periods': 12
	}

	fit_params = {
		'smoothing_level': 0.3,
		'smoothing_trend': 0.2,
		'smoothing_seasonal': 0.1,
		'optimized': True
	}

	wrapper_params = {
		'init__trend': 'add',
		'init__damped_trend': False,
		'init__seasonal': 'add',
		'init__seasonal_periods': 12,
		'fit__smoothing_level': 0.3,
		'fit__smoothing_trend': 0.2,
		'fit__smoothing_seasonal': 0.1,
		'fit__optimized': True
	}

	sm_model = ExponentialSmoothing(endog=synthetic_series, **init_params)
	sm_model_fit = sm_model.fit(**fit_params)
	sm_predictions = sm_model_fit.predict(start=prediction_series.index[0], end=prediction_series.index[-1])

	wrapper_model = StatsmodelsWrapper(model_class=ExponentialSmoothing, **wrapper_params)
	wrapper_model.fit(synthetic_series)
	wrapper_model_predictions = wrapper_model.predict(prediction_series)


	assert wrapper_model_predictions.equals(sm_predictions)

def test_initialize_sm_wrapper_parameters_using_series(synthetic_series, prediction_series):
	init_params = {
		'trend': 'add',
		'damped_trend': False,
		'seasonal': 'add',
		'seasonal_periods': 12
	}

	fit_params = {
		'smoothing_level': 0.3,
		'smoothing_trend': 0.2,
		'smoothing_seasonal': 0.1,
		'optimized': True
	}


	wrapper_params = {
		'init__trend': 'add',
		'init__damped_trend': False,
		'init__seasonal': 'add',
		'init__seasonal_periods': 12,
		'fit__smoothing_level': 0.3,
		'fit__smoothing_trend': 0.2,
		'fit__smoothing_seasonal': 0.1,
		'fit__optimized': True
	}


	sm_model = ExponentialSmoothing(endog=synthetic_series, **init_params)
	sm_model_fit = sm_model.fit(**fit_params)
	sm_predictions = sm_model_fit.predict(start=prediction_series.index[0], end=prediction_series.index[-1])

	wrapper_model = StatsmodelsWrapper(
		model_class=ExponentialSmoothing,
		init_params=init_params,
		fit_params=fit_params
	)

	wrapper_model.fit(synthetic_series)
	wrapper_model_predictions = wrapper_model.predict(prediction_series)

	assert wrapper_model_predictions.equals(sm_predictions)

def test_update_sm_wrapper_parameters_as_kwargs_using_series(synthetic_series, prediction_series):
	init_params = {
		'trend': 'add',
		'damped_trend': False,
		'seasonal': 'add',
		'seasonal_periods': 12
	}

	fit_params = {
		'smoothing_level': 0.3,
		'smoothing_trend': 0.2,
		'smoothing_seasonal': 0.1,
		'optimized': True
	}

	starting_wrapper_params = {
		'init__trend': 'add',
		'init__damped_trend': False,
		'fit__smoothing_trend': 0.9,
		'fit__smoothing_seasonal': 0.1,
		'fit__optimized': True,
	}

	updated_wrapper_params = {
		'init__seasonal': 'add',
		'init__seasonal_periods': 12,
		'fit__smoothing_level': 0.3,
		'fit__smoothing_trend': 0.2,
		'model_class': ExponentialSmoothing
	}

	sm_model = ExponentialSmoothing(endog=synthetic_series, **init_params)
	sm_model_fit = sm_model.fit(**fit_params)
	sm_predictions = sm_model_fit.predict(start=prediction_series.index[0], end=prediction_series.index[-1])

	wrapper_model = StatsmodelsWrapper(**starting_wrapper_params)
	wrapper_model.set_params(**updated_wrapper_params)

	wrapper_model.fit(synthetic_series)
	wrapper_model_predictions = wrapper_model.predict(prediction_series)

	assert wrapper_model_predictions.equals(sm_predictions)


@pytest.mark.parametrize(
	('selected_column', 'unselected_column'),
	[
		('endog', 'mock_column'),
		('mock_column', 'endog')
	]
)
def test_initialize_sm_wrapper_parameters_as_kwargs_using_dataframe(synthetic_dataframe, prediction_dataframe, selected_column, unselected_column):
	init_params = {
		'trend': 'add',
		'damped_trend': False,
		'seasonal': 'add',
		'seasonal_periods': 12
	}

	fit_params = {
		'smoothing_level': 0.3,
		'smoothing_trend': 0.2,
		'smoothing_seasonal': 0.1,
		'optimized': True
	}

	wrapper_params = {
		'init__trend': 'add',
		'init__damped_trend': False,
		'init__seasonal': 'add',
		'init__seasonal_periods': 12,
		'fit__smoothing_level': 0.3,
		'fit__smoothing_trend': 0.2,
		'fit__smoothing_seasonal': 0.1,
		'fit__optimized': True
	}

	selected_series = synthetic_dataframe[selected_column]

	sm_model = ExponentialSmoothing(endog=selected_series, **init_params)
	sm_model_fit = sm_model.fit(**fit_params)
	sm_predictions = sm_model_fit.predict(start=prediction_dataframe.index[0], end=prediction_dataframe.index[-1])

	wrapper_model = StatsmodelsWrapper(
		model_class=ExponentialSmoothing,
		target_column=selected_column,
		**wrapper_params
	)

	wrapper_model.fit(synthetic_dataframe)
	wrapper_model_predictions = wrapper_model.predict(prediction_dataframe)


	#TODO use assert_frame_equal
	assert wrapper_model_predictions[unselected_column].equals(prediction_dataframe[unselected_column])
	assert wrapper_model_predictions[selected_column].equals(sm_predictions)

# TODO
# Add other test cases but for dataframe

