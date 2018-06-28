#data preprocessing

#importing libraries

import numpy as np #mathematical tools

import pandas as pd  #best library, import and manage data sets

import matplotlib.pyplot as plt


dataset= pd.read_csv('data.csv')

# matrix of features ( independent variables)

x = dataset.iloc[:, :-1].values #colon means we take all the columns, while colon -1  we dont take last column
y = dataset.iloc[:, 3].values

#take care of missing data

# we use scikit learn preprocessing library for this , import imputer class

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#CATEGORICAL DATA

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
x
x[: , 0 ] = labelencoder_x.fit_transform(x[: , 0 ])

#dummy variable

onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
x
#label encoder is used for dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting data set into training and test set
from sklearn.model_selection import train_test_split  
x_train, x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2 , random_state = 0)
x

# feature scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

