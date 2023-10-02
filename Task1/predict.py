#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle


# In[3]:


ptest_df = pd.read_csv('mars-private_test-reg.csv')
ptest_df.head()


# In[4]:


test_columns = ['№ испытания','Модуль сигнала','Тип_измерения','Количество импульсов','Фаза Hor',
                 'Фаза Ver','Уровень шума','Азимут','У.М.','Секунда','Дальность (м)','Доля сигнала в ВП']


# In[5]:


columns = ['id_test', 'signal_module', 'measure_type', 'impulce_amount',
              'fase_hor', 'fase_ver', 'noise_level', 'azimut', 'U.M.',
              'second', 'distance_m', 'signal_share']
ptest_df.columns = columns[:-1]


# In[7]:


xgb_model_loaded = pickle.load(open('xgb_reg.pkl', "rb"))


# In[8]:


result = ptest_df.copy()
ptest_df = pd.get_dummies(ptest_df, columns = ['id_test','measure_type','impulce_amount'], dtype=float)
result['signal_share'] = xgb_model_loaded.predict(ptest_df)


# In[9]:


result.loc[result['signal_share']>1, 'signal_share'] = 1
result.loc[result['signal_share']<0, 'signal_share'] = 0


# In[10]:


result.columns = test_columns
result.head()


# In[65]:


result.to_csv('xgboost_gpu_15k_reg.csv', index=False)

