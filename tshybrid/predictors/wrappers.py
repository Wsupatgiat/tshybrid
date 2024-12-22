from sklearn.base import BaseEstimator, RegressorMixin

class StatsmodelsWrapper(BaseEstimator, RegressorMixin):
	def __init__(self, model_class, **parameters):
		self.model_class = model_class
		self.parameters = parameters

		print(self.parameters)
		self.init_params = dict()
		self.fit_params = dict()


	def set_params(self, **parameters):

		for parameter, value in parameters.items():
			if parameter.startswith('init__'):
				self.init_params.update({parameter.lstrip('init__'): value})

			elif parameter.startswith('fit__'):
				self.init_params.update({parameter.lstrip('fit__'): value})

			else:
				setattr(self, parameter, value)

		return self

	def fit(self, X, y=None):
		self.model = self.model_class(endog=X, **self.init_params)
		self.model_fit = self.model.fit(**self.fit_params)

	def predict(self, X):
		predictions = self.model_fit.predict(start=X.index[0], end=X.index[-1])
		return predictions

