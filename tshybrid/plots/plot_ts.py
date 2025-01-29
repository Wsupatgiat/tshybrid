import pandas as pd
import matplotlib.pyplot as plt

def plot_df_vert(df, figsize=(10, 3)):
	num_columns = len(df.columns)
	fig, axes = plt.subplots(num_columns, 1, figsize=(figsize[0], figsize[1] * num_columns), sharex=True)

	if num_columns == 1:
		axes = [axes]

	for i, col in enumerate(df.columns):
		axes[i].plot(df.index, df[col], label=col)
		axes[i].set_title(col)
		axes[i].legend()

