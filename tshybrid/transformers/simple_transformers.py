from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from tshybrid.base.base_class import BaseTimeSeriesProcessor

class ShiftMin(BaseEstimator, TransformerMixin, BaseTimeSeriesProcessor):
	'''
	shift the time series to a certain value (desired_min)
	'''

	def __init__(self, target_column='endog', desired_min=1):
		self.target_column = target_column
		self.desired_min = desired_min

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


class RootTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, target_column='endog', degree=2):
		self.target_column = target_column
		self.degree = degree

	def fit(self, X, y=None):
		if isinstance(X, pd.Series):
			sel_X = X
		elif isinstance(X, pd.DataFrame):
			sel_X = X[self.target_column]
		else:
			#TODO add test case for this
			raise TypeError(f"Expected either a Dataframe or Series type, but got {type(X)}")

		if (sel_X < 0).any() and self.degree % 2 == 0:
			raise ValueError("Series contain negative numbers, which is incompatible with even degree roots")

		if self.degree < 2:
			raise ValueError(f"Expected degree higher than 1, but got {self.degree}")

		return self

	def transform(self, X):
		if isinstance(X, pd.Series):
			sel_X = X
		elif isinstance(X, pd.DataFrame):
			sel_X = X[self.target_column]
		else:
			#TODO add test case for this
			raise TypeError(f"Expected either a Dataframe or Series type, but got {type(X)}")

		transformed_series = sel_X**(1/self.degree)

		if isinstance(X, pd.Series):
			return transformed_series
		else:
			X[self.target_column] = transformed_series
			return X
		


	def inverse_transform(self, X):
		if isinstance(X, pd.Series):
			sel_X = X
		elif isinstance(X, pd.DataFrame):
			sel_X = X[self.target_column]
		else:
			#TODO add test case for this
			raise TypeError(f"Expected either a Dataframe or Series type, but got {type(X)}")

		original_series = sel_X**self.degree

		if isinstance(X, pd.Series):
			return original_series
		else:
			X[self.target_column] = original_series
			return X




