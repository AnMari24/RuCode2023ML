#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle


# In[2]:


df = pd.read_csv('mars-train-regr.csv')
df.head()


# In[6]:


df.columns = ['id_test', 'signal_module', 'measure_type', 'impulce_amount',
              'fase_hor', 'fase_ver', 'noise_level', 'azimut', 'U.M.',
              'second', 'distance_m', 'signal_share']


# In[12]:


df1 = df.copy()


# In[13]:


df1 = pd.get_dummies(df1, columns = ['id_test','measure_type','impulce_amount'], dtype=float)
df1.head()


# In[54]:


bst = XGBRegressor(
    objective = "reg:squarederror",
    n_estimators = 15000,
    max_depth = 10,
    learning_rate = 0.01,
    random_state = 77,
)
bst.fit(df1.drop(['signal_share'], axis=1), df1['signal_share'])


# In[ ]:


file_name = "xgb_reg.pkl"
pickle.dump(bst, open(file_name, "wb"))

