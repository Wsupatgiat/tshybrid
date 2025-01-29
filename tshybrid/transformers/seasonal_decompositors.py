import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from statsmodels.tsa.seasonal import STL

from tshybrid.base.base_class import BaseTimeSeriesProcessor

class STLDecompose(BaseTimeSeriesProcessor, TransformerMixin, BaseEstimator):
	'''
	inverse always return a dataframe
	maybe fix this later?
	'''
	def __init__(
		self, target_column='endog',
		trend_column='trend',
		season_column='season',
		residuals_column='residuals',
		#INIT
		period=None,
		seasonal=7,
		trend=None,
		low_pass=None,
		seasonal_deg=1,
		trend_deg=1,
		low_pass_deg=1,
		robust=False,
		seasonal_jump=1,
		trend_jump=1,
		low_pass_jump=1,
		#FIT
		inner_iter=None,
		outer_iter=None
	):
		self.trend_column = trend_column
		self.season_column = season_column
		self.residuals_column = residuals_column

		self.period = period
		self.seasonal = seasonal
		self.trend = trend
		self.low_pass = low_pass
		self.seasonal_deg = seasonal_deg
		self.trend_deg = trend_deg
		self.low_pass_deg = low_pass_deg
		self.robust = robust
		self.seasonal_jump = seasonal_jump
		self.trend_jump = trend_jump
		self.low_pass_jump = low_pass_jump

		self.inner_iter = inner_iter
		self.outer_iter = outer_iter

		super().__init__(target_column=target_column)

	def fit(self, X, y=None):
		sel_X = self._select_series(X)
		self.stl_class = STL(
			sel_X,
			period = self.period,
			seasonal = self.seasonal,
			trend = self.trend,
			low_pass = self.low_pass,
			seasonal_deg = self.seasonal_deg,
			trend_deg = self.trend_deg,
			low_pass_deg = self.low_pass_deg,
			robust = self.robust,
			seasonal_jump = self.seasonal_jump,
			trend_jump = self.trend_jump,
			low_pass_jump = self.low_pass_jump
		)

		self.stl_fit = self.stl_class.fit(inner_iter=self.inner_iter, outer_iter=self.outer_iter)

		self.is_fitted_ = True
		return self

	def transform(self, X):
		check_is_fitted(self)

		X = X.copy()
		sel_X = self._select_series(X)

		seasonal_decomposed = pd.DataFrame({
			self.trend_column: self.stl_fit.trend,
			self.season_column: self.stl_fit.seasonal,
			self.residuals_column: self.stl_fit.resid
		})

		return self._replace_series_df(X, seasonal_decomposed)

	def inverse_transform(self, X):
		check_is_fitted(self)

		X = X.copy()
		dropped_columns = [self.trend_column, self.season_column, self.residuals_column]

		original_series = X[dropped_columns].sum(axis=1)

		return self._replace_df_series(X, original_series, dropped_columns)


