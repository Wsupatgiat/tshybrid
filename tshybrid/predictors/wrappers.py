from sklearn.base import BaseEstimator, RegressorMixin

class StatsmodelsWrapper(BaseEstimator, RegressorMixin):
	def __init__(self, model_class=None, init_params=None, fit_params=None, **kwargs):
		self.model_class = model_class

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

		self.model = self.model_class(endog=X, **self.init_params)
		self.model_fit = self.model.fit(**self.fit_params)

	def predict(self, X):
		predictions = self.model_fit.predict(start=X.index[0], end=X.index[-1])
		return predictions

