import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tshybrid.base.base_class import BaseTimeSeriesProcessor


class StatsmodelsWrapper(BaseEstimator, RegressorMixin, BaseTimeSeriesProcessor):
	def __init__(self, target_column='endog', model_class=None, init_params=None, fit_params=None, **kwargs):
		#TODO check model class added
		self.model_class = model_class
		self.target_column = target_column

		self.init_params = init_params if init_params is not None else {}
		self.fit_params = fit_params if fit_params is not None else {}
		self.kwargs = kwargs


	def _parse_kwargs(self):
		for parameter, value in self.kwargs.items():
			if parameter.startswith('init__'):
				self.init_params.update({parameter[6:]: value})

			elif parameter.startswith('fit__'):
				self.fit_params.update({parameter[5:]: value})
			else:
				setattr(self, parameter, value)

		return self

	def set_params(self, **parameters):
		self.kwargs.update(parameters)
		self._parse_kwargs()
		return self

	def fit(self, X, y=None):
		self._parse_kwargs()

		sel_X = self._select_series(X)

		self.model = self.model_class(endog=sel_X, **self.init_params)
		self.model_fit = self.model.fit(**self.fit_params)

	#TODO FIX RETURN COL
	def predict(self, X):
		sel_X = self._select_series(X)

		predictions = self.model_fit.predict(start=sel_X.index[0], end=sel_X.index[-1])

		return self._replace_series(X, predictions)
