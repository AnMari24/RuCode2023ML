#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle


# In[44]:


df = pd.read_csv('mars-train-class.csv')
df.head()


# In[48]:


df.columns = ['id_test', 'signal_module', 'measure_type', 'impulce_amount',
              'fase_hor', 'fase_ver', 'noise_level', 'azimut', 'U.M.',
              'second', 'distance_m', 'signal_share', 'mars_type']


# In[54]:


df1 = df.copy()


# In[55]:


df1 = pd.get_dummies(df1, columns = ['id_test','measure_type','impulce_amount'], dtype=float)
df1.head()


# In[80]:


bst = XGBClassifier(
    objective = 'binary:logistic',
    n_estimators = 15000,
    max_depth = 10,
    learning_rate = 0.01,
    random_state = 77,
)
bst.fit(df1.drop(['mars_type'], axis=1), df1['mars_type'])


# In[ ]:


file_name = "xgb_class.pkl"
pickle.dump(bst, open(file_name, "wb"))

