#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_adv_data = pd.read_csv('C:\\Users\\Nikhil\\Documents\\Datasets\\Advertising.csv')


# In[3]:


df_adv_data.head()


# In[4]:


df_adv_data.size


# In[5]:


df_adv_data.shape


# In[6]:


df_adv_data.columns


# In[7]:


X_feature = df_adv_data[['TV', 'Radio', 'Newspaper']]


# In[8]:


X_feature.head()


# In[9]:


Y_target = df_adv_data[['Sales']]


# In[10]:


Y_target.head()


# In[11]:


X_feature.shape


# In[12]:


Y_target.shape


# In[35]:


pip install scikit-learn


# In[14]:


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X_feature,Y_target,random_state=1)


# In[16]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[17]:


#Linear regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train,y_train)


# In[18]:


linreg


# In[20]:


print(linreg.intercept_)
print(linreg.coef_)


# In[21]:


y_pred = linreg.predict(x_test)
y_pred


# In[22]:


from sklearn import metrics
import numpy as np


# In[24]:


print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:




