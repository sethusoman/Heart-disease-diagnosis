import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#for multiclass dataset
#dataset  = pd.read_csv("combined_dataset.csv.csv")

#for binaryclass dataset
dataset  = pd.read_csv("recons_dataset/combined_dataset.csv")
#x = dataset.iloc[:,0:13].values
#y = dataset.iloc[:,13].values

classifier = RandomForestClassifier()
scaler = StandardScaler()
avgAcc = [0] * 100

for i in range(100):
	#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)


	scaler = MinMaxScaler()
	train, test = train_test_split(dataset, test_size=0.25)

	X_train = train.drop('num', axis=1)
	Y_train = train['num']

	X_test = test.drop('num', axis=1)
	Y_test = test['num']



	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)

	classifier.fit(X_train, Y_train)
	predicted = classifier.predict(X_test)
	acc = accuracy_score(Y_test, predicted)
	print("---------Train = ",i+1," --------------")
	print("Orginal label	:\n",Y_test)
	print("Predicted value	:\n",predicted)
	print("Accuracy : ",acc*100, "%")
	avgAcc[i] = acc
print(avgAcc)
print("Average accuracy after 100 iteration : ",sum(avgAcc))
f = open("accuracy/Randomforest-multi-Result.txt","w")
for i in range(len(avgAcc)) :
	f.write("%f \n"%avgAcc[i])

f.write("Average accuracy : %f"% sum(avgAcc))
f.close()
