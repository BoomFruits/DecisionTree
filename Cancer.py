import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from TreeNode import DecisionTreeID3

df = pd.read_csv('Cancer_Data.csv') 
le = preprocessing.LabelEncoder()
data = df.apply(le.fit_transform)
dt_Train,dt_Test = train_test_split(data,test_size=0.3)
X_train = dt_Train.iloc[:,:-1]
y_train = dt_Train.iloc[:,-1]
X_test = dt_Test.iloc[:,:-1]
tree = DecisionTreeID3(max_depth=10,min_samples_split=3)
tree.fit(X_train,y_train)
# print(tree.predict(X_test))