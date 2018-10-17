#Breast Cancer Prediction
#Importing the required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the data
data=pd.read_csv('data_2.csv')
#Checking if missing data is present or not
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()  

#Converting the categorical variable to dummy variable(0 or !)
diag=pd.get_dummies(data['diagnosis'],drop_first=True)
data.drop(['diagnosis','id'],axis=1,inplace=True)
data=pd.concat([data,diag],axis=1)
#PCA to reduce the features
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
variance=pca.explained_variance_ratio_

X=data.iloc[: ,0:30]
y=data.iloc[:, 31:32]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Using Gaussian Naive Bayes model to train the model
'''from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)'''

#Using Logistic Regression to predict the model

from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(X_train,y_train)
y2_pred=lg.predict(X_test)

#Predicting the result
predict=lg.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
a=accuracy_score(y2_pred,y_test)  #0.93.7%
cm=confusion_matrix(y_test,y2_pred)
'''array([[84,  6],
       [ 3, 50]], dtype=int64)'''
