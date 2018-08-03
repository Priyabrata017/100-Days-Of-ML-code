import numpy as np
import pandas as pd

dataset=pd.read_csv('Social_Network_Ads.csv')
dataset.head()
dataset.drop('User ID',axis=1,inplace=True)
Sex=pd.get_dummies(dataset['Gender'],drop_first=True)
dataset=pd.concat([dataset,Sex],axis=1)
dataset.drop('Gender',axis=1,inplace=True)
predictors=['Age' , 'EstimatedSalary' ,  'Male']
outcomes=['Purchased']
x=dataset[predictors]
y=dataset[outcomes]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset = sc.fit_transform(x)
dataset = sc.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
predict=clf.predict(X_test)
from sklearn.metrics import accuracy_score
a=accuracy_score(predict,y_test)
    

