import pandas as pd
import numpy as np
import pandas_profiling as pp

df=pd.read_csv("/Users/Shravani/OneDrive/Desktop/DATA/admission1.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())
df=df.drop("Research",axis=1)
df=df.drop("Unnamed: 0",axis=1)
print(df.info())

#profile report
#profile=pp.ProfileReport(df)
#profile.to_file("/Users/Shravani/OneDrive/Desktop/DATA/admission1EDA.html")
from sklearn.model_selection import train_test_split
y=df['Chance_of_Admit']
X=df.drop('Chance_of_Admit', axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=1)

from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
print(regression_model.fit(X_train, y_train))
intercept = regression_model.intercept_[None]
print(intercept)
for idx,col_name in enumerate(X_train.columns):
    print("The co-effecient for {} is {} ".format(col_name,regression_model.coef_[idx]))
