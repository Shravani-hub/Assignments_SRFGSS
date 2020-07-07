import numpy as np
import pandas as pd
import pandas_profiling as pp
df=pd.read_csv("/Users/Shravani/OneDrive/Desktop/DATA/admission1.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())
df=df.drop("Research",axis=1)
df=df.drop("Unnamed: 0",axis=1)
print(df.info)
#profile=pp.profile_report
#profile.to_file("/Users/Shravani/OneDrive/Desktop/DATA/admission1EDA.html")

from sklearn.model_selection import train_test_split
y=df['Chance_of_Admit']
x=df.drop('Chance_of_Admit',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
y_pred=knn.predict((x_test))
print(x_test)
print(y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score
conf_mat=confusion_matrix(y_test,y_pred)
print(conf_mat)
acc_score=accuracy_score(y_test,y_pred)
print(acc_score)

#pre_deployment test
y_pred_new=knn.predict([[310,106,2,3.5,3.6,7.0]])
print(y_pred_new)
