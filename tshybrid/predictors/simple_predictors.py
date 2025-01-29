import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from tshybrid.base.base_class import BaseTimeSeriesProcessor

class Linear(BaseTimeSeriesProcessor, RegressorMixin, BaseEstimator):
	def __init__(self, target_column='endog'):
		super().__init__(target_column=target_column)

	def fit(self, X, y=None):
		sel_X = self._select_series(X)

		self.is_fitted_ = True
		return self

	#TODO FIX RETURN COL
	def predict(self, X):
		check_is_fitted(self)

		sel_X = self._select_series(X)


		return self._replace_series(X, predictions)
