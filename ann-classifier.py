import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

#for multiclass dataset
#dataframe = pd.read_csv("cleveland-normalize.csv", header=0)
#for binaryclass dataset
dataframe = pd.read_csv("datasets/cleveland-normalize-binary.csv", header=0)
dataset = dataframe.values
X = dataset[:,0:13].astype(float)
Y = dataset[:,13]

#encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#conver integers to dummy variables (one hot encoded)
hotEncoded_y = np_utils.to_categorical(encoded_Y)

avgAcc = [0] * 100
for i in range(100):
	x_train, x_test, y_train,y_test = train_test_split(X,hotEncoded_y, test_size=0.2)
	#create model
	"""
	<multiclass>

	model = Sequential()
	model.add(Dense(200, input_dim=13, activation = 'sigmoid'))
	model.add(Dense(75, activation = 'sigmoid'))
	model.add(Dense(5, activation = 'softmax'))
	#compile model
	model.compile(loss = 'categorical_crossentropy',optimizer= 'adam', metrics=['categorical_accuracy'])
	#fit the dataset to our model	
	model.fit(x=x_train, y=y_train, epochs=300, validation_data = (x_test, y_test))	

	<!multiclass>
	"""
	#<binaryclass>
	model = Sequential()
	model.add(Dense(18, input_dim=13, activation = 'sigmoid'))
	model.add(Dense(2, activation = 'sigmoid'))
	#compile model
	model.compile(loss = 'binary_crossentropy',optimizer='adam', metrics=['accuracy'])
	#fit the dataset to our model
	model.fit(x=x_train, y=y_train, epochs=600, validation_data = (x_test, y_test))
	#<!binaryclass>
	loss,acc= model.evaluate(x_test, y_test)

	predict = model.predict(x_test)
	original_labels = np.argmax(y_test, axis = 1)
	prediction = np.argmax(predict, axis = 1)
	accuracy = accuracy_score(original_labels, prediction)
	
	print("Original labels  : \n",original_labels)
	print("Predictions	: \n",prediction)
	print("Accuracy result  : \n",accuracy*100, " %")
	avgAcc[i] = accuracy

print(avgAcc)
print("Average accuracy after 100 iterations : ", sum(avgAcc))
f = open("accuracy/annClassifier-Binary-Result.txt","w")
for i in range(len(avgAcc)) :
	f.write("%f \n"%avgAcc[i])
f.write("Average accuracy : %f"% sum(avgAcc))
f.close()


