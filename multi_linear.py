#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:49:07 2018

@author: sarojbehera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Import the Preprocessing Module from sklearn
# Onehot Encoder would be need to transform your Categorical Column
 # to numerical so that the ML algo can be run on
 # e.g. City New York as 1, Florida to 2 etc 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable Not required in this case 
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

#Avoiding dummy variable trap
X= X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling will be taken care by Library
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
# Fit your training data into the Model
regressor.fit(X_train,y_train)

#predict off the Test data 
y_predictions= regressor.predict(X_test)

# predicted profits with real predicts

# import stats model
import statsmodels.formula.api as sm

#X = np.append(arr= X,values= np.ones((50,1)).astype(int),axis=1)

# add the new coulumns of 1 to begining and not append at tehe end
# because stats model requires it, y = mx + c ( here x0 is counted)
X = np.append(arr=  np.ones((50,1)).astype(int),values=X,axis=1)
# Significance Value - SL - 0.05
# if P > SL then stay 
# list the index for independent variable
X_opt=X[:,[0,1,2,3,4,5]]
# step2 for Backward elimination
# ols - ordinary Least Square Models 
# Interceptor is not included by default , hence we include '1s' Columns in it.
  ## refer formula -> y = b0 + b1x1 + b2x2 ..... bnxn .
#endog= dependent variable, exdog=array of num of obs and k is num of regressors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# step4 .. look for stats Sl and P value
  #from  statsmodels.iolib import summary as sum
  #regressor_sum=sum.summary(regressor_OLS)

# Remove variable 2 as P value is greater than SL and check the predictions
X_opt=X[:,[0,1,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# Continue to check for  P value if greater than SL and check the predictions
# Next remove the variable and run again

X_opt=X[:,[0,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Continue to check for  P value if greater than SL and check the predictions
# Next remove the variable and run again
X_opt=X[:,[0,3,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# if 6% remove it but you can keep it 
# If you runbelow,  you will notice that "R&D spend" columns can predict the better PROFIT 
X_opt=X[:,[0,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


