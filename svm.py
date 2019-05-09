import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

data = pd.read_csv("datasets/cleveland-normalize-binary.csv")

X = data.drop('num', axis=1)
Y = data['num']

X = X.astype('float64')
Y = Y.astype('float64')

scaler = StandardScaler()
svclassifier = SVC(kernel='linear')

avgAcc = [0]*100

for i in range(100) :
	x_train, x_test, y_train,y_test = train_test_split(X,Y, test_size=0.2)
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	
	svclassifier.fit(x_train,y_train)
	predicted = svclassifier.predict(x_test)
	acc = accuracy_score(y_test, predicted)
	print("---------Train = ",i+1," --------------")
	print("Support Vector Machine : \n")
	print("original label : \n", y_test.values)
	print("predicted : \n",predicted)
	print("Accuracy : ", acc*100, " %")
	avgAcc[i] = acc

#print('Average accuracy after 100 iterations : ', avgAcc*10,' %')

print(avgAcc)
print(sum(avgAcc))
f = open("accuracy/svm-Result.txt","w")
for i in range(len(avgAcc)) :
	f.write("%f \n"%avgAcc[i])


f.write("Average accuracy : %f"% sum(avgAcc))
f.close()

