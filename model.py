from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

root = os.path.dirname(__file__)
path_df = os.path.join(root, 'recons_dataset/combined_dataset1.csv')
data = pd.read_csv(path_df)
nlenx=20
acc = 0
acc_binary = 0
scaler = StandardScaler()
train, test = train_test_split(data, test_size=0.2)

X_train = train.drop('num', axis=1)
Y_train = train['num']

X_test = test.drop('num', axis=1)
Y_test = test['num']

# We don't scale targets: Y_test, Y_train as SVC returns the class labels not probability values
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
#clf = LogisticRegression(multi_class='ovr')
clf = RandomForestClassifier(n_estimators=20, random_state=0)
#clf=SVC(kernel='linear')
# Training the classifier
clf.fit(X_train, Y_train)

Y_hat = clf.predict(X_test)
Y_hat_bin = Y_hat>0
Y_test_bin = Y_test>0
acc = acc + accuracy_score(Y_hat, Y_test)
acc_binary = acc_binary +accuracy_score(Y_hat_bin, Y_test_bin)
# Testing model accuracy. Average is taken as test set is very small hence accuracy varies a lot everytime the model is trained
#acc = 0
#acc_binary = 0
nlenX=20
for i in range(0, nlenX):
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    #clf=SVC(kernel='linear')
    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    clf.fit(X_train, Y_train)
    Y_hat = clf.predict(X_test)
    Y_hat_bin = Y_hat>0
    Y_test_bin = Y_test>0
    acc = acc + accuracy_score(Y_hat, Y_test)
    acc_binary = acc_binary +accuracy_score(Y_hat_bin, Y_test_bin)


#print("Average test Accuracy:{}".format(acc/nlenx))
#print(" Confusion matrix \n", confusion_matrix(Y_test, Y_hat_bin))
print("Average test accuracy:{}".format(acc_binary/nlenx))

# Saving the trained model for inference
model_path = os.path.join(root, 'model/rfc.sav')
joblib.dump(clf, model_path)

# Saving the scaler object
scaler_path = os.path.join(root, 'model/scaler.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
