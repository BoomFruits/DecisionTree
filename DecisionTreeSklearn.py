import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
df = pd.read_csv('Cancer_Data.csv')
le = preprocessing.LabelEncoder()
data = df.apply(le.fit_transform)
train_data,test_data = train_test_split(data,test_size=0.3)
X_train = train_data.drop(["diagnosis"],axis = 1)
y_train = train_data["diagnosis"]
X_test = test_data.drop(["diagnosis"],axis = 1)
y_test = test_data["diagnosis"]
from sklearn.tree import DecisionTreeClassifier #Cay su dung cho bai toan phan lop
# from sklearn.metrics import confusion
my_tree = DecisionTreeClassifier(max_depth=11)
my_tree.fit(X_train,y_train)
#du doan tren du lieu test
y_pred = my_tree.predict(X_test)
print("Ty le du doan dung: ",accuracy_score(y_pred,y_test))


