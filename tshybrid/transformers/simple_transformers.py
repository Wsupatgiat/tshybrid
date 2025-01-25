from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from tshybrid.base.base_class import BaseTimeSeriesProcessor

class ShiftMin(BaseTimeSeriesProcessor, BaseEstimator, TransformerMixin):
	'''
	shift the time series to a certain value (desired_min)
	'''

	def __init__(self, target_column='endog', desired_min=1):
		self.desired_min = desired_min

		super().__init__(target_column=target_column)



	def fit(self, X, y=None):
		sel_X = self._select_series(X)
		self.min_value = np.min(sel_X)

		return self

	def transform(self, X):
		X = X.copy()
		sel_X = self._select_series(X)
		shifted_series = sel_X - self.min_value + self.desired_min

		return self._replace_series(X, shifted_series)

	def inverse_transform(self, X):
		X = X.copy()
		sel_X = self._select_series(X)
		original_series = sel_X + self.min_value - self.desired_min

		return self._replace_series(X, original_series)

class RootTransformer(BaseTimeSeriesProcessor, BaseEstimator, TransformerMixin):
	def __init__(self, target_column='endog', degree=2):
		self.degree = degree

		super().__init__(target_column=target_column)

	def fit(self, X, y=None):
		sel_X = self._select_series(X)

		if (sel_X < 0).any() and self.degree % 2 == 0:
			raise ValueError("Series contain negative numbers, which is incompatible with even degree roots")

		if self.degree < 2:
			raise ValueError(f"Expected degree higher than 1, but got {self.degree}")

		return self

	def transform(self, X):
		X = X.copy()
		sel_X = self._select_series(X)

		transformed_series = sel_X**(1/self.degree)

		return self._replace_series(X, transformed_series)

	def inverse_transform(self, X):
		X = X.copy()
		sel_X = self._select_series(X)
		original_series = sel_X**self.degree

		return self._replace_series(X, original_series)




