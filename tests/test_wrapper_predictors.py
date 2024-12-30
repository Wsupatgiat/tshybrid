import pytest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tshybrid.predictors.wrappers import StatsmodelsWrapper


def test_initialize_sm_wrapper_parameters_as_kwargs(monthly_seasonal_synthetic_series):
	synthetic_series, prediction_series = monthly_seasonal_synthetic_series

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

def test_initialize_sm_wrapper_parameters(monthly_seasonal_synthetic_series):
	synthetic_series, prediction_series = monthly_seasonal_synthetic_series

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

def test_update_sm_wrapper_parameters_as_kwargs(monthly_seasonal_synthetic_series):
	synthetic_series, prediction_series = monthly_seasonal_synthetic_series

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
