

# # Linear Regresssion with Boston Housing Data 1970


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


# In[6]:


#get_ipython().magic('matplotlib inline')


# In[7]:


# use existing dataset provided by sklearn library
boston = load_boston()


# In[10]:


boston.keys()


# In[11]:


boston['DESCR']


# In[12]:


boston['data']


# In[13]:


print(boston.feature_names)


# In[16]:



df=pd.DataFrame(boston.data)


# In[17]:


df.head()


# In[18]:


df.columns=('CRIM' ,'ZN', 'INDUS' ,'CHAS' ,'NOX' ,'RM', 'AGE' ,'DIS', 'RAD', 'TAX' ,'PTRATIO','B' ,'LSTAT')

# OR easy way  :)
# bos.columns = boston.feature_names


# In[20]:


df['PRICE']=boston.target


# In[23]:


df.head()


# In[29]:


df.describe()


# In[24]:


from sklearn.linear_model import LinearRegression


# In[30]:


# Below deprecated
#from sklearn.cross_validation import train_test_split

# Use this rather
from sklearn.model_selection import train_test_split


# In[48]:


x=df.drop('PRICE',axis=1)
y=df['PRICE']


# In[49]:


#Split your data into train and test 
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.4,random_state=101)


# In[50]:


lm=LinearRegression()


# In[51]:


lm.fit(x_train,y_train)


# In[52]:


print(lm.intercept_)


# In[54]:


coef=pd.DataFrame(lm.coef_,x.columns,columns=['coefficient'])


# In[55]:


coef


# In[56]:


predictions=lm.predict(x_test)


# In[57]:


predictions


# In[41]:


lm.coef_


# In[58]:


plt.scatter(y_test,predictions)


# In[63]:


sns.distplot((y_test-predictions),bins=50)


# In[67]:


#Mean Squared Error
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,predictions)
mse


# In[66]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

