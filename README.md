# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: INDHUMATHI L
RegisterNumber:  212224220037

```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```
```
print("data.info():")
data.info()
```
```
print("isnull() and sum():")
data.isnull().sum()
```
```
print("data value counts():")
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

```
## Output:

<img width="1583" height="362" alt="image" src="https://github.com/user-attachments/assets/44fbc2f7-a6a2-490d-befe-2d96e5989e68" />

<img width="652" height="416" alt="image" src="https://github.com/user-attachments/assets/d9d69ba9-d414-4fbf-884d-7460ba0c3b27" />

<img width="306" height="514" alt="image" src="https://github.com/user-attachments/assets/3aee22ed-8cde-4c12-a642-3cc7849bb5f1" />

<img width="278" height="255" alt="image" src="https://github.com/user-attachments/assets/73b45f73-17e3-429a-b2d0-8a7bf9b89b40" />

<img width="1576" height="284" alt="image" src="https://github.com/user-attachments/assets/78ec9112-e9f5-44fa-90b4-6b3bfa5d98a5" />

<img width="1390" height="282" alt="image" src="https://github.com/user-attachments/assets/e2a442f4-f509-4471-96ea-1e70a854184b" />

<img width="296" height="82" alt="image" src="https://github.com/user-attachments/assets/c1ee7bfe-fd49-4158-ab01-c128cb24c4cf" />

<img width="1735" height="98" alt="image" src="https://github.com/user-attachments/assets/b9fa2789-14a6-4c2d-8863-ec50ba8b5365" />

<img width="915" height="639" alt="image" src="https://github.com/user-attachments/assets/06be1110-d9ca-4777-96c1-7d00dcfb1ed1" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
