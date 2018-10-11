
# coding: utf-8

# # Linear Regresssion with Boston Housing Data 1970
# Below isconverted from Jupyter notebook ,so would work best in that
# There are some Exploratory Data Visualization in between as well.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# if you want to run in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# use existing dataset provided by sklearn library
boston = load_boston()

# its basically as dictionary so Check the keys
boston.keys()

boston['DESCR']

boston['data']

#Check the features identified
print(boston.feature_names)

df=pd.DataFrame(boston.data)

df.head()


df.columns=('CRIM' ,'ZN', 'INDUS' ,'CHAS' ,'NOX' ,'RM', 'AGE' ,'DIS', 'RAD', 'TAX' ,'PTRATIO',
 'B' ,'LSTAT')

# OR easy way  :)
# bos.columns = boston.feature_names


df['PRICE']=boston.target

df.head()
df.describe()

# import LinearRegresion library
from sklearn.linear_model import LinearRegression

# Below deprecated
#from sklearn.cross_validation import train_test_split

# Use this rather
from sklearn.model_selection import train_test_split



x=df.drop('PRICE',axis=1)
y=df['PRICE']


#Split your data into train and test 
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.4,random_state=101)


lm=LinearRegression()

# Fit the training data to the Model
lm.fit(x_train,y_train)

print(lm.intercept_)

coef=pd.DataFrame(lm.coef_,x.columns,columns=['coefficient'])

coef

predictions=lm.predict(x_test)

predictions

lm.coef_

#PLot a scattered plot to view the TEST data against the predicted output
plt.scatter(y_test,predictions)

sns.distplot((y_test-predictions),bins=50)


#Mean Squared Error
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,predictions)
mse



print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

