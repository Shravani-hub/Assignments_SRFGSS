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
print(df.info)
#profile=pp.profile_report
#profile.to_file("/Users/Shravani/OneDrive/Desktop/DATA/admission1EDA.html")

from sklearn.model_selection import train_test_split
y=df['Chance_of_Admit']
x=df.drop('Chance_of_Admit',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)

from sklearn.linear_model import LogisticRegression
regression_model=LogisticRegression
regression_model = LogisticRegression()
regression_model.fit(x_train,y_train)
intercept = regression_model.intercept_[0]
print(intercept)
for idx,col_name in enumerate(x_train.columns):
    print("The coeffecient of {} is {}".format(col_name,regression_model.coef_[0][idx]))

