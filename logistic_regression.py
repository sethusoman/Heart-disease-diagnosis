#logistic regression
import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = pd.read_csv("datasets/cleveland-normalize-binary.csv")
#weight = {0:0.54,1:0.18,2:0.12,3:0.12,4:0.04}
logReg = LogisticRegression(multi_class='ovr')
avgAcc = [0] * 100

def cross_validate_plot(data, logReg):
	data = data.astype('float64')
	predicted = cross_val_score(logReg,data.iloc[:,0:13], data['num'],cv=100)	
	print(predicted)
	print("Average accuracy after 100 iterations : ",sum(predicted), "%")
	plt.plot(predicted)
	plt.axis([0,100,0,1.00])
	plt.ylabel('accuracy')
	plt.show()
#Logistic Regression with Monte-carlo validation
def LogRes_MCCross_Validation(data, logReg, avgAcc):		
	for i in range(100) : 
		x_train, x_test= train_test_split(data, test_size=0.2)
		x_train = x_train.astype('float64')
		x_testLabel = x_test['num'].values
		x_test = x_test.drop('num', 1)
		logReg.fit(x_train.iloc[:,0:13], x_train['num'])
		predicted =  logReg.predict(x_test)
		acc = accuracy_score(x_testLabel, predicted)
		print("---------Train = ",i+1," --------------")
		print("original label : \n",x_testLabel)
		print("predicted : \n",logReg.predict(x_test))
		print("Accuracy : ", acc*100, "%")
		avgAcc[i] = acc

	print(avgAcc)
	print("Average accuracy after 100 iterations : ",sum(avgAcc), "%")
	plt.plot(avgAcc)
	plt.axis([0,100,0,1.00])
	plt.ylabel('accuracy')
	plt.show()


LogRes_MCCross_Validation(data,logReg,avgAcc)


#monte_carlo(data,logReg,avgAcc)

f = open("accuracy/knn-Result.txt","w")
for i in range(len(avgAcc)) :
	f.write("%f \n"%avgAcc[i])

f.write("Average accuracy : %f"% sum(avgAcc))
f.close()
"""
def parallel_cor(data):
	plt.figure(figsize=(20,10))
	parallel_coordinates(data,'num')
	plt.show()
	
def histogram(data):
	data.hist(bins=15,color='steelblue', edgecolor= 'black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
	plt.tight_layout(rect=(0,0,1,1))
	plt.show()
"""
