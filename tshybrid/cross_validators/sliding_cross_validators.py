import numpy as np
from sklearn.model_selection import BaseCrossValidator

class SlidingWindowCV(BaseCrossValidator):
	'''
	sliding window validation (with steps = 1)
	'''
	def __init__(self, train_size=None, horizon=None):
		self.train_size = train_size
		self.horizon = horizon

	def get_n_splits(self, X, y=None, groups=None):
		return len(X) - self.train_size - self.horizon + 1

	def split(self, X, y=None):
		for start in range(0, len(X) - self.train_size - self.horizon + 1):
			train_idx = np.arange(start, start + self.train_size)
			test_idx = np.arange(start + self.train_size, start + self.train_size + self.horizon)
			yield train_idx, test_idx





