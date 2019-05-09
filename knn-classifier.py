import pandas as pd
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#only for multiclass
#import data
data = pd.read_csv("datasets/cleveland-normalize.csv")
neigh = KNeighborsClassifier(n_neighbors=3)
avgAcc = [0] * 100

for i in range(100) : 
	x_train, x_test= train_test_split(data, test_size=0.2)
	x_train = x_train.astype['float64']
	x_testLabel = x_test['num'].values
	x_test = x_test.drop('num', 1)
	neigh.fit(x_train.iloc[:,0:13], x_train['num'])
	predicted =  neigh.predict(x_test)
	acc = accuracy_score(x_testLabel, predicted)
	
	print("---------Train = ",i+1," --------------")
	print("original label : \n",x_testLabel)
	print("predicted : \n",neigh.predict(x_test))
	print("Accuracy : ", acc*100, "%")
	avgAcc[i] = acc

print(avgAcc)
print("Average accuracy after 100 iterations : ",sum(avgAcc), "%")
f = open("accuracy/knn-Result.txt","w")
for i in range(len(avgAcc)) :
	f.write("%f \n"%avgAcc[i])

f.write("Average accuracy : %f"% sum(avgAcc))
f.close()


#data = pd.read_csv("cleveland-normalize-binary.csv")
