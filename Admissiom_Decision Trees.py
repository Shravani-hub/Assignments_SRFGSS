import pandas as pd
import numpy as np
import pandas_profiling as pp
df=pd.read_csv("/Users/Shravani/OneDrive/Desktop/DATA/admission1.csv")
print(df.head())
print(df.dtypes)
print(df.shape)
df=df.drop("Research",axis=1)
df=df.drop("Unnamed: 0",axis=1)
print(df.info())
#profile = pp.ProfileReport(df)
#profile.to_file("/Users/rajaathota72/Desktop/reportadmission.html")
y = df['Chance_of_Admit']
X = df.drop('Chance_of_Admit',axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
# evaluation metrics
from sklearn.metrics import accuracy_score,confusion_matrix
conf_mat = confusion_matrix(y_pred,y_test)
acc_score = accuracy_score(y_pred,y_test)
print(conf_mat)
print(acc_score)
#Pre_deployment
y_pred_new=model.predict([[310,106,2,3.5,3.6,7.0]])
print(y_pred_new)
