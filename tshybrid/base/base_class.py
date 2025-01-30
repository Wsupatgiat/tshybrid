import pandas as pd

class BaseTimeSeriesProcessor:
	def __init__(self, target_column='endog'):
		self.target_column = target_column

	'''
	handles and validates the correct input (pandas series or dataframe)
	'''
	def _initialize_fit_state(self, X):
		self._fitted_type = type(X)
		self._freq = X.index.freq or X.index.inferred_freq
		if self._freq is None:
			raise ValueError("No frequency found")

		return self

	def _select_series(self, X):
		if isinstance(X, pd.Series):
			return X
		elif isinstance(X, pd.DataFrame):
			return X[self.target_column]
		else:
			raise TypeError(f"Expected either a DataFrame or Series type, got {type(X)}")

	def _replace_series(self, X, new_X):
		if isinstance(X, pd.Series):
			return new_X
		elif isinstance(X, pd.DataFrame):
			X[self.target_column] = new_X
			return X
		else:
			raise TypeError(f"Expected either a DataFrame or Series type, got {type(X)}")

	def _replace_series_df(self, X, new_X):
		if isinstance(X, pd.Series):
			return new_X
		elif isinstance(X, pd.DataFrame):
			X = X.drop(columns=[self.target_column])
			X = pd.merge(X, new_X, left_index=True, right_index=True)
			return X
		else:
			raise TypeError(f"Expected either a DataFrame or Series type, got {type(X)}")

	def _replace_df_series(self, X, new_X, dropped_columns):
		if self._fitted_type == pd.Series:
			return new_X
		elif self._fitted_type == pd.DataFrame:
			X[self.target_column] = new_X
			X = X.drop(columns=dropped_columns)
			return X
		else:
			raise TypeError(f"Expected either a DataFrame, got {type(X)}")
