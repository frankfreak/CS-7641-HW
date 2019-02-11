import numpy as np
import pandas as pd
import time
import pickle

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.tree._tree import TREE_LEAF
from sklearn.ensemble import AdaBoostClassifier

######################################### Reading and Splitting the Data ###############################################
data = pd.read_csv('spambase.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

#Split dataset into training set and test set randomly (test set amount ranging from 90% -10% )
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(x_data, y_data, shuffle=True, test_size=0.1, random_state=random_state)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(x_data, y_data, shuffle=True, test_size=0.2, random_state=random_state)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(x_data, y_data, shuffle=True, test_size=0.3, random_state=random_state)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(x_data, y_data, shuffle=True, test_size=0.4, random_state=random_state)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(x_data, y_data, shuffle=True, test_size=0.5, random_state=random_state)
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(x_data, y_data, shuffle=True, test_size=0.6, random_state=random_state)
X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(x_data, y_data, shuffle=True, test_size=0.7, random_state=random_state)
X_train_8, X_test_8, y_train_8, y_test_8 = train_test_split(x_data, y_data, shuffle=True, test_size=0.8, random_state=random_state)
X_train_9, X_test_9, y_train_9, y_test_9 = train_test_split(x_data, y_data, shuffle=True, test_size=0.9, random_state=random_state)

# ############################################### Decision Tree ###############################################
# Decision Tree with different training sizes
dt1 = DecisionTreeClassifier(max_depth=5).fit (X_train_1, y_train_1)
dt2 = DecisionTreeClassifier(max_depth=5).fit (X_train_2, y_train_2)
dt3 = DecisionTreeClassifier(max_depth=5).fit (X_train_3, y_train_3)
dt4 = DecisionTreeClassifier(max_depth=5).fit (X_train_4, y_train_4)
dt5 = DecisionTreeClassifier(max_depth=5).fit (X_train_5, y_train_5)
dt6 = DecisionTreeClassifier(max_depth=5).fit (X_train_6, y_train_6)
dt7 = DecisionTreeClassifier(max_depth=5).fit (X_train_7, y_train_7)
dt8 = DecisionTreeClassifier(max_depth=5).fit (X_train_8, y_train_8)
dt9 = DecisionTreeClassifier(max_depth=5).fit (X_train_9, y_train_9)
tree.export_graphviz(dt1, out_file='tree1.dot')
tree.export_graphviz(dt2, out_file='tree2.dot')
tree.export_graphviz(dt3, out_file='tree3.dot')
tree.export_graphviz(dt4, out_file='tree4.dot')
tree.export_graphviz(dt5, out_file='tree5.dot')
tree.export_graphviz(dt6, out_file='tree6.dot')
tree.export_graphviz(dt7, out_file='tree7.dot')
tree.export_graphviz(dt8, out_file='tree8.dot')
tree.export_graphviz(dt9, out_file='tree9.dot')

print("train set: 90% ,d = 5")
print("{:.0%}".format(accuracy_score(y_train_1, dt1.predict(X_train_1).round())))
print("{:.0%}".format(accuracy_score(y_test_1, dt1.predict(X_test_1).round())))
print("train set: 80% ,d = 5")
print("{:.0%}".format(accuracy_score(y_train_2, dt2.predict(X_train_2).round())))
print("{:.0%}".format(accuracy_score(y_test_2, dt2.predict(X_test_2).round())))
print("train set: 70% ,d = 5")
print("{:.0%}".format(accuracy_score(y_train_3, dt3.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, dt3.predict(X_test_3).round())))
print("train set: 60% ,d = 5")
print("{:.0%}".format(accuracy_score(y_train_4, dt4.predict(X_train_4).round())))
print("{:.0%}".format(accuracy_score(y_test_4, dt4.predict(X_test_4).round())))
print("train set: 50% ,d = 5")
print("{:.0%}".format(accuracy_score(y_train_5, dt5.predict(X_train_5).round())))
print("{:.0%}".format(accuracy_score(y_test_5, dt5.predict(X_test_5).round())))
print("train set: 40% ,d = 5")
print("{:.0%}".format(accuracy_score(y_train_6, dt6.predict(X_train_6).round())))
print("{:.0%}".format(accuracy_score(y_test_6, dt6.predict(X_test_6).round())))
print("train set: 30% ,d = 5")
print("{:.0%}".format(accuracy_score(y_train_7, dt7.predict(X_train_7).round())))
print("{:.0%}".format(accuracy_score(y_test_7, dt7.predict(X_test_7).round())))
print("train set: 20% ,d = 5")
print("{:.0%}".format(accuracy_score(y_train_8, dt8.predict(X_train_8).round())))
print("{:.0%}".format(accuracy_score(y_test_8, dt8.predict(X_test_8).round())))
print("train set: 10% ,d = 5")
print("{:.0%}".format(accuracy_score(y_train_9, dt9.predict(X_train_9).round())))
print("{:.0%}".format(accuracy_score(y_test_9, dt9.predict(X_test_9).round())))

# dt with d (20 - 1) & 70% training set
for x in xrange(1, 21):
	dt = DecisionTreeClassifier(max_depth=x).fit (X_train_3, y_train_3)
	print("depth = " + str(x))
	print("{:.0%}".format(accuracy_score(y_train_3, dt.predict(X_train_3).round())))
 	print("{:.0%}".format(accuracy_score(y_test_3, dt.predict(X_test_3).round())))

# ############################################### Multi Layer Perceptron #################################################
mlp1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_1, y_train_1)
mlp2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_2, y_train_2)
mlp3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_3, y_train_3)
mlp4 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_4, y_train_4)
mlp5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_5, y_train_5)
mlp6 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_6, y_train_6)
mlp7 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_7, y_train_7)
mlp8 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_8, y_train_8)
mlp9 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_9, y_train_9)


mlp10 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_3, y_train_3)
mlp11 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_3, y_train_3)
mlp12 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_3, y_train_3)

mlp13 = MLPClassifier(activation='identity', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_3, y_train_3)
mlp14 = MLPClassifier(activation='logistic', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_3, y_train_3)
mlp15 = MLPClassifier(activation='tanh', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_3, y_train_3)
mlp16 = MLPClassifier(activation='relu', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train_3, y_train_3)
print("train set: 90%")
print("{:.0%}".format(accuracy_score(y_train_1, mlp1.predict(X_train_1).round())))
print("{:.0%}".format(accuracy_score(y_test_1, mlp1.predict(X_test_1).round())))
print("train set: 80%")
print("{:.0%}".format(accuracy_score(y_train_2, mlp2.predict(X_train_2).round())))
print("{:.0%}".format(accuracy_score(y_test_2, mlp2.predict(X_test_2).round())))
print("train set: 70%")
print("{:.0%}".format(accuracy_score(y_train_3, mlp3.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, mlp3.predict(X_test_3).round())))
print("train set: 60%")
print("{:.0%}".format(accuracy_score(y_train_4, mlp4.predict(X_train_4).round())))
print("{:.0%}".format(accuracy_score(y_test_4, mlp4.predict(X_test_4).round())))
print("train set: 50%")
print("{:.0%}".format(accuracy_score(y_train_5, mlp5.predict(X_train_5).round())))
print("{:.0%}".format(accuracy_score(y_test_5, mlp5.predict(X_test_5).round())))
print("train set: 40%")
print("{:.0%}".format(accuracy_score(y_train_6, mlp6.predict(X_train_6).round())))
print("{:.0%}".format(accuracy_score(y_test_6, mlp6.predict(X_test_6).round())))
print("train set: 30%")
print("{:.0%}".format(accuracy_score(y_train_7, mlp7.predict(X_train_7).round())))
print("{:.0%}".format(accuracy_score(y_test_7, mlp7.predict(X_test_7).round())))
print("train set: 20%")
print("{:.0%}".format(accuracy_score(y_train_8, mlp8.predict(X_train_8).round())))
print("{:.0%}".format(accuracy_score(y_test_8, mlp8.predict(X_test_8).round())))
print("train set: 10%")
print("{:.0%}".format(accuracy_score(y_train_9, mlp9.predict(X_train_9).round())))
print("{:.0%}".format(accuracy_score(y_test_9, mlp9.predict(X_test_9).round())))

print("{:.0%}".format(accuracy_score(y_train_3, mlp10.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, mlp10.predict(X_test_3).round())))
print("{:.0%}".format(accuracy_score(y_train_3, mlp11.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, mlp11.predict(X_test_3).round())))
print("{:.0%}".format(accuracy_score(y_train_3, mlp12.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, mlp12.predict(X_test_3).round())))

print("{:.0%}".format(accuracy_score(y_train_3, mlp13.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, mlp13.predict(X_test_3).round())))
print("{:.0%}".format(accuracy_score(y_train_3, mlp14.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, mlp14.predict(X_test_3).round())))
print("{:.0%}".format(accuracy_score(y_train_3, mlp15.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, mlp15.predict(X_test_3).round())))
print("{:.0%}".format(accuracy_score(y_train_3, mlp16.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, mlp16.predict(X_test_3).round())))

# ############################################### AdaBoost ###############################################
# ab with different training size 
ab1 = AdaBoostClassifier(n_estimators=50).fit (X_train_1, y_train_1)
ab2 = AdaBoostClassifier(n_estimators=50).fit (X_train_2, y_train_2)
ab3 = AdaBoostClassifier(n_estimators=50).fit (X_train_3, y_train_3)
ab4 = AdaBoostClassifier(n_estimators=50).fit (X_train_4, y_train_4)
ab5 = AdaBoostClassifier(n_estimators=50).fit (X_train_5, y_train_5)
ab6 = AdaBoostClassifier(n_estimators=50).fit (X_train_6, y_train_6)
ab7 = AdaBoostClassifier(n_estimators=50).fit (X_train_7, y_train_7)
ab8 = AdaBoostClassifier(n_estimators=50).fit (X_train_8, y_train_8)
ab9 = AdaBoostClassifier(n_estimators=50).fit (X_train_9, y_train_9)
print("train set: 90%")
print("{:.0%}".format(accuracy_score(y_train_1, ab1.predict(X_train_1).round())))
print("{:.0%}".format(accuracy_score(y_test_1, ab1.predict(X_test_1).round())))
print("train set: 80%")
print("{:.0%}".format(accuracy_score(y_train_2, ab2.predict(X_train_2).round())))
print("{:.0%}".format(accuracy_score(y_test_2, ab2.predict(X_test_2).round())))
print("train set: 70%")
print("{:.0%}".format(accuracy_score(y_train_3, ab3.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, ab3.predict(X_test_3).round())))
print("train set: 60%")
print("{:.0%}".format(accuracy_score(y_train_4, ab4.predict(X_train_4).round())))
print("{:.0%}".format(accuracy_score(y_test_4, ab4.predict(X_test_4).round())))
print("train set: 50%")
print("{:.0%}".format(accuracy_score(y_train_5, ab5.predict(X_train_5).round())))
print("{:.0%}".format(accuracy_score(y_test_5, ab5.predict(X_test_5).round())))
print("train set: 40%")
print("{:.0%}".format(accuracy_score(y_train_6, ab6.predict(X_train_6).round())))
print("{:.0%}".format(accuracy_score(y_test_6, ab6.predict(X_test_6).round())))
print("train set: 30%")
print("{:.0%}".format(accuracy_score(y_train_7, ab7.predict(X_train_7).round())))
print("{:.0%}".format(accuracy_score(y_test_7, ab7.predict(X_test_7).round())))
print("train set: 20%")
print("{:.0%}".format(accuracy_score(y_train_8, ab8.predict(X_train_8).round())))
print("{:.0%}".format(accuracy_score(y_test_8, ab8.predict(X_test_8).round())))
print("train set: 10%")
print("{:.0%}".format(accuracy_score(y_train_9, ab9.predict(X_train_9).round())))
print("{:.0%}".format(accuracy_score(y_test_9, ab9.predict(X_test_9).round())))

# ab with n_estimators (1 - 100) & 70% training set
x = 1
while x <= 200:
	ab = AdaBoostClassifier(n_estimators=x).fit (X_train_3, y_train_3)
	print("n_estimators = " + str(x))
	print("{:.0%}".format(accuracy_score(y_train_3, ab.predict(X_train_3).round())))
	print("{:.0%}".format(accuracy_score(y_test_3, ab.predict(X_test_3).round())))
	x = x + 10


# ############################################### k-nearest neighbors ###############################################
# kNN with different training size & k = 5
kNN1 = KNeighborsClassifier(n_neighbors=5).fit (X_train_1, y_train_1)
kNN2 = KNeighborsClassifier(n_neighbors=5).fit (X_train_2, y_train_2)
kNN3 = KNeighborsClassifier(n_neighbors=5).fit (X_train_3, y_train_3)
kNN4 = KNeighborsClassifier(n_neighbors=5).fit (X_train_4, y_train_4)
kNN5 = KNeighborsClassifier(n_neighbors=5).fit (X_train_5, y_train_5)
kNN6 = KNeighborsClassifier(n_neighbors=5).fit (X_train_6, y_train_6)
kNN7 = KNeighborsClassifier(n_neighbors=5).fit (X_train_7, y_train_7)
kNN8 = KNeighborsClassifier(n_neighbors=5).fit (X_train_8, y_train_8)
kNN9 = KNeighborsClassifier(n_neighbors=5).fit (X_train_9, y_train_9)
print("train set: 90% ,k = 5")
print("{:.0%}".format(accuracy_score(y_train_1, kNN1.predict(X_train_1).round())))
print("{:.0%}".format(accuracy_score(y_test_1, kNN1.predict(X_test_1).round())))
print("train set: 80% ,k = 5")
print("{:.0%}".format(accuracy_score(y_train_2, kNN2.predict(X_train_2).round())))
print("{:.0%}".format(accuracy_score(y_test_2, kNN2.predict(X_test_2).round())))
print("train set: 70% ,k = 5")
print("{:.0%}".format(accuracy_score(y_train_3, kNN3.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, kNN3.predict(X_test_3).round())))
print("train set: 60% ,k = 5")
print("{:.0%}".format(accuracy_score(y_train_4, kNN4.predict(X_train_4).round())))
print("{:.0%}".format(accuracy_score(y_test_4, kNN4.predict(X_test_4).round())))
print("train set: 50% ,k = 5")
print("{:.0%}".format(accuracy_score(y_train_5, kNN5.predict(X_train_5).round())))
print("{:.0%}".format(accuracy_score(y_test_5, kNN5.predict(X_test_5).round())))
print("train set: 40% ,k = 5")
print("{:.0%}".format(accuracy_score(y_train_6, kNN6.predict(X_train_6).round())))
print("{:.0%}".format(accuracy_score(y_test_6, kNN6.predict(X_test_6).round())))
print("train set: 30% ,k = 5")
print("{:.0%}".format(accuracy_score(y_train_7, kNN7.predict(X_train_7).round())))
print("{:.0%}".format(accuracy_score(y_test_7, kNN7.predict(X_test_7).round())))
print("train set: 20% ,k = 5")
print("{:.0%}".format(accuracy_score(y_train_8, kNN8.predict(X_train_8).round())))
print("{:.0%}".format(accuracy_score(y_test_8, kNN8.predict(X_test_8).round())))
print("train set: 10% ,k = 5")
print("{:.0%}".format(accuracy_score(y_train_9, kNN9.predict(X_train_9).round())))
print("{:.0%}".format(accuracy_score(y_test_9, kNN9.predict(X_test_9).round())))

# kNN with k (20 - 1) & 70% training set
for x in xrange(1, 21):
	kNN = KNeighborsClassifier(n_neighbors=x).fit (X_train_3, y_train_3)
	print("k = " + str(x))
	print("{:.0%}".format(accuracy_score(y_train_3, kNN.predict(X_train_3).round())))
	print("{:.0%}".format(accuracy_score(y_test_3, kNN.predict(X_test_3).round())))

# ############################################ Support Vector Machine ###################################################

# scaler = StandardScaler()
# scaler.fit(X_train_3)
# X_train = scaler.transform(X_train_3)
# scaler.fit(X_test_3)
# X_test = scaler.transform(X_test_3)

svc1 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_1, y_train_1)
svc2 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_2, y_train_2)
svc3 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_3, y_train_3)
svc4 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_4, y_train_4)
svc5 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_5, y_train_5)
svc6 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_6, y_train_6)
svc7 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_7, y_train_7)
svc8 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_8, y_train_8)
svc9 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_9, y_train_9)
svc10 = SVC(kernel = 'rbf', gamma='auto').fit(X_train_3, y_train_3)


print("train set: 90%")
print("{:.0%}".format(accuracy_score(y_train_1, svc1.predict(X_train_1).round())))
print("{:.0%}".format(accuracy_score(y_test_1, svc1.predict(X_test_1).round())))
print("train set: 80%")
print("{:.0%}".format(accuracy_score(y_train_2, svc2.predict(X_train_2).round())))
print("{:.0%}".format(accuracy_score(y_test_2, svc2.predict(X_test_2).round())))
print("train set: 70%")
print("{:.0%}".format(accuracy_score(y_train_3, svc3.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, svc3.predict(X_test_3).round())))
print("train set: 60%")
print("{:.0%}".format(accuracy_score(y_train_4, svc4.predict(X_train_4).round())))
print("{:.0%}".format(accuracy_score(y_test_4, svc4.predict(X_test_4).round())))
print("train set: 50%")
print("{:.0%}".format(accuracy_score(y_train_5, svc5.predict(X_train_5).round())))
print("{:.0%}".format(accuracy_score(y_test_5, svc5.predict(X_test_5).round())))
print("train set: 40%")
print("{:.0%}".format(accuracy_score(y_train_6, svc6.predict(X_train_6).round())))
print("{:.0%}".format(accuracy_score(y_test_6, svc6.predict(X_test_6).round())))
print("train set: 30%")
print("{:.0%}".format(accuracy_score(y_train_7, svc7.predict(X_train_7).round())))
print("{:.0%}".format(accuracy_score(y_test_7, svc7.predict(X_test_7).round())))
print("train set: 20%")
print("{:.0%}".format(accuracy_score(y_train_8, svc8.predict(X_train_8).round())))
print("{:.0%}".format(accuracy_score(y_test_8, svc8.predict(X_test_8).round())))
print("train set: 10%")
print("{:.0%}".format(accuracy_score(y_train_9, svc9.predict(X_train_9).round())))
print("{:.0%}".format(accuracy_score(y_test_9, svc9.predict(X_test_9).round())))
print("{:.0%}".format(accuracy_score(y_train_3, svc10.predict(X_train_3).round())))
print("{:.0%}".format(accuracy_score(y_test_3, svc10.predict(X_test_3).round())))
