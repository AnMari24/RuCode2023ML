#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[2]:


ptest_df = pd.read_csv('mars-public_test-class.csv')
ptest_df.head()


# In[4]:


test_columns = ['№ испытания','Модуль сигнала','Тип_измерения','Количество импульсов','Фаза Hor','Фаза Ver',
                 'Уровень шума','Азимут','У.М.','Секунда','Дальность (м)','Доля сигнала в ВП','Тип марсианина']


# In[96]:


columns = ['id_test', 'signal_module', 'measure_type', 'impulce_amount',
              'fase_hor', 'fase_ver', 'noise_level', 'azimut', 'U.M.',
              'second', 'distance_m', 'signal_share', 'mars_type']
ptest_df.columns = columns[:-1]


# In[97]:


result = ptest_df.copy()
ptest_df = pd.get_dummies(ptest_df, columns = ['id_test','measure_type','impulce_amount'], dtype=float)
result['mars_type'] = bst.predict(ptest_df)


# In[102]:


result.columns = test_columns
result.head()


# In[103]:


result.to_csv('xgboost_gpu_15k_all.csv', index=False)

