import pandas as pd
import numpy as np
import seaborn as sns
data = pd.read_csv(r"C:\Users\VIKRANT TOMAR\Anaconda3\creditcard\dataset.csv")
data.tail()
fraud = data.loc[data['Class'] == 1]
normal = data.loc[data['Class'] == 0]
len(fraud)
len(normal)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
X = data.iloc[:,:-1]
y= data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.35)
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X_train, y_train)
y_pred = np.array(clf.predict(X_test))
y = np.array(y_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))