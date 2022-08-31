#!/usr/bin/env python
# coding: utf-8

# # CALIFORNIA MEDIAN HOUSE VALUE PREDICTION USING REGRESSION
# **IMPORTING IMPORTANT LIBRARIES**

# In[1]:


import numpy as np
import pandas as pd
import sklearn.datasets


# **FETCHING CALIFORNIA DATASET FROM SCIKIT LEARN** 

# In[2]:


from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)


# In[3]:


california_housing.frame.head()


# In[4]:


calhouse=pd.DataFrame(california_housing.data, columns = california_housing.feature_names)


# In[5]:


calhouse.head()


# In[6]:


calhouse['MedHouseVal']=california_housing.target


# In[7]:


calhouse.head()


# # TARGET IS MedHouseVal #

# In[40]:


calhouse.isna().sum()


# In[41]:


calhouse.info()


# In[8]:


calhouse.shape


# In[9]:


calhouse.isnull().sum()


# In[10]:


calhouse.describe()


# **ANALYZING CORRELATION**

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[50]:


sns.pairplot(calhouse)


# In[12]:


plt.figure(figsize=(16,9))
sns.heatmap(calhouse.corr(), cbar=True,  annot=True, annot_kws={'size':8}, cmap='summer')


# In[13]:


X = calhouse.drop(['MedHouseVal'], axis=1)
Y = calhouse['MedHouseVal']


# In[14]:


X.head()


# In[15]:


Y.head()


# **SPLITTING DATASET INTO TRAINING AND TEST DATA**

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# **FEATURE SCALING**

# In[18]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)


# ## MODEL SELECTION ##
# **LINEAR REGRESSION**

# In[19]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)


# In[20]:


lr.coef_


# In[21]:


lr.intercept_


# In[47]:


lr_pred=lr.predict(X_test)
lr_pred


# In[48]:


lr_pred[:1]


# In[23]:


lr.score(X_test,Y_test)


# **ANALYZING METRICS SCORE**

# In[24]:


from sklearn import metrics


# In[49]:


r_square_error = metrics.r2_score(Y_test, lr_pred)
mean_absolute_error = metrics.mean_absolute_error(Y_test, lr_pred)

print("R squared error : ",r_square_error)
print('Mean Absolute Error : ',mean_absolute_error)


# **RIDGE REGRESSION**

# In[26]:


from sklearn.linear_model import Ridge,Lasso


# In[27]:


rd=Ridge()
rd.fit(X_train,Y_train)
rd.score(X_test,Y_test)


# In[43]:


rd_pred=rd.predict(X_test)
rd_pred


# In[45]:


rd_pred[:1]


# **LASSO REGRESSION**

# In[29]:


ls=Lasso()
ls.fit(X_train,Y_train)
ls.score(X_test,Y_test)


# In[30]:


ls.predict(X_test)


# In[57]:


ls.predict(X_test[:1])


# **XGBOOST REGRESSION**

# In[31]:


from xgboost import XGBRegressor


# In[32]:


XGB = XGBRegressor()


# In[33]:


XGB.fit(X_train, Y_train)


# In[37]:


XGB.predict(X_test)


# In[35]:


XGB.score(X_test, Y_test)


# In[56]:


XGB.predict(X_test[:1])


# **METRICS SCORE**

# In[58]:


r_square_error = metrics.r2_score(Y_test,XGB.predict(X_test) )
mean_absolute_error = metrics.mean_absolute_error(Y_test, XGB.predict(X_test))

print("R squared error : ",r_square_error)
print('Mean Absolute Error : ',mean_absolute_error)


# # OEDER OF REGRESSION MODEL:XGBoost>LINEAR >RIDGE>LASSO #

# In[ ]:




