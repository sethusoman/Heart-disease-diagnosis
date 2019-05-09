import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("datasets/cleveland-normalize-binary.csv")

x = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

avgAcc = [0] * 100

for i in range(100) : 
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
	dtree_model = DecisionTreeClassifier(max_depth=2).fit(x_train,y_train)
	predicted = dtree_model.predict(x_test)
	acc = accuracy_score(y_test, predicted)
	
	print("Decision Tree Classifier : ")
	print("Original label: \n",y_test)
	print("Predicted : \n",predicted)
	print("Accuracy : \n",acc*100,"%")
	avgAcc[i] = acc
print(avgAcc)
print("Average accuracy after 100 iterations : ",sum(avgAcc))
f = open("accuracy/DecisionTreeResult.txt","w")
for i in range(len(avgAcc)) :
	f.write("%f \n"%avgAcc[i])

f.write("Average accuracy : %f"% sum(avgAcc))
f.close()



