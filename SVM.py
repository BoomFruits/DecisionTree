import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
df = pd.read_csv('.vscode/cars.csv') 
X_data = np.array(df[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']].values)

def data_encoder(X): 
    for i, j in enumerate(X):
        for k in range(0, 7): 
            if (j[k] == "vhigh"): 
                j[k] = 0 
            elif (j[k] == "high"):
                j[k] = 1
            elif (j[k] == "med"):
                j[k] = 2
            elif (j[k] == "low"):
                j[k] = 3
            elif (j[k] == "2"):
                j[k] = 4
            elif (j[k] == "3"):
                j[k] = 5
            elif (j[k] == "4"):
                j[k] = 6
            elif (j[k] == "5more"):
                j[k] = 7
            elif (j[k] == "more"):
                j[k] = 8
            elif (j[k] == "small"):
                j[k] = 9
            elif (j[k] == "big"):
                j[k] = 10
            # elif (j[k] == "acc"):
            #     j[k] = 1
            # elif (j[k] == "unacc"):
            #     j[k] = -1
    return X
data=data_encoder(X_data)
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = True) 

X_train = dt_Train[:, :6] 
y_train = dt_Train[:, 6] 
X_test = dt_Test[:, :6] 
y_test = dt_Test[:, 6]
svr = SVC(gamma="auto")
svr.fit(X_train, y_train) 
y_predict = svr.predict(X_test) 
acc = accuracy_score(y_predict,y_test)
print('Ty le du doan dung la: ',acc)