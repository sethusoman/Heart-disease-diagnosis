#Random forest classifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from sklearn.externals import joblib
import pickle


#for multiclass dataset
#dataset  = pd.read_csv("cleveland-normalize.csv")

#for binaryclass dataset
root = os.path.dirname(__file__)
dataset  = pd.read_csv("datasets/cleveland-normalize-binary.csv")
x = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

classifier = RandomForestClassifier()
scaler = StandardScaler()
avgAcc = [0] * 100

for i in range(100):
	x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.fit_transform(x_test)

	classifier.fit(x_train, y_train)
	predicted = classifier.predict(x_test)
	acc = accuracy_score(y_test, predicted)
	print("---------Train = ",i+1," --------------")
	print("Orginal label	:\n",y_test)
	print("Predicted value	:\n",predicted)
	print("Accuracy : ",acc*100, "%")
	avgAcc[i] = acc
print(avgAcc)
print("Average accuracy after 100 iteration : ",sum(avgAcc))
#f = open("accuracy/Randomforest-multi-Result.txt","w")
#for i in range(len(avgAcc)) :
#	f.write("%f \n"%avgAcc[i])

#f.write("Average accuracy : %f"% sum(avgAcc))
#f.close()

# Saving the trained model for inference
model_path = os.path.join(root, 'rfc.sav')
joblib.dump(classifier, model_path)

# Saving the scaler object
scaler_path = os.path.join(root, 'scaler.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
