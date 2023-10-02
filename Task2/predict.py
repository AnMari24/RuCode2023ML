#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle


# In[12]:


ptest_df = pd.read_csv('mars-private_test-class.csv')
ptest_df.head()


# In[13]:


test_columns = ['№ испытания','Модуль сигнала','Тип_измерения','Количество импульсов','Фаза Hor','Фаза Ver',
                 'Уровень шума','Азимут','У.М.','Секунда','Дальность (м)','Доля сигнала в ВП','Тип марсианина']


# In[14]:


columns = ['id_test', 'signal_module', 'measure_type', 'impulce_amount',
              'fase_hor', 'fase_ver', 'noise_level', 'azimut', 'U.M.',
              'second', 'distance_m', 'signal_share', 'mars_type']
ptest_df.columns = columns[:-1]


# In[15]:


xgb_model_loaded = pickle.load(open('xgb_class.pkl', "rb"))


# In[16]:


result = ptest_df.copy()
ptest_df = pd.get_dummies(ptest_df, columns = ['id_test','measure_type','impulce_amount'], dtype=float)
result['mars_type'] = xgb_model_loaded.predict(ptest_df)


# In[17]:


result.columns = test_columns
result.head()


# In[18]:


result.to_csv('xgboost_gpu_15k_class.csv', index=False)

