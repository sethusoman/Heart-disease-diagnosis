import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("recons_dataset/combined_dataset.csv")

#histogram for x = feature, y = number of data belonging to what category in that feature
def histogram(data):
	data.hist(bins=15,color='steelblue', edgecolor= 'black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
	plt.tight_layout(rect=(0,0,1,1))
	plt.show()

# visualize heatmap correlation between attribute
def heatmap(data):
	f,ax = plt.subplots(figsize=(10,6))
	corr = data.corr()
	hm = sns.heatmap(round(corr,2),annot=True, ax=ax, cmap="coolwarm", fmt='.2f',linewidths=.05)
	f.subplots_adjust(top=0.93)
	t = f.suptitle('Heart Disease Attributes Correlation Heatmap',fontsize=14)
	plt.show()

#pair-wise scatter plots
def pairwise(data):
	cols = list(data)
	pp = sns.pairplot(data[cols],height=1.8, aspect=1.8, plot_kws=dict(edgecolor="k",linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))
	fig = pp.fig
	fig.subplots_adjust(top=0.93, wspace = 0.3)
	t = fig.suptitle('Heart Disease Pairwise Plots', fontsize=14)
	plt.show()

#parallel coordinates
def parcordinate(data):
	cols = list(data)
	subset = data[cols]
	from sklearn.preprocessing import StandardScaler
	ss = StandardScaler()
	scaled_df = ss.fit_transform(subset)
	scaled_df = pd.DataFrame(scaled_df, columns=cols)
	final = pd.concat([scaled_df, data['num']],axis=1)
	plt.figure(figsize=(20,10))
	pc = parallel_coordinates(final, 'num')
	plt.show()


heatmap(data)


#heatmap(data)
