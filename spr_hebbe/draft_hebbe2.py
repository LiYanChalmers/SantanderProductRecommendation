#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:01:58 2016

@author: lyaa

This requires large memory, can only run on cluster
"""

from santanderStart import *

#%%
print('loading and sorting')
df = pd.read_csv('train_clean.csv')
dt = pd.read_csv('test_clean.csv')
ntrain = df.shape[0]
ntest = dt.shape[0]
df = pd.concat([df, dt])
df = df.sort_values(by=['ncodpers', 'fecha_dato'], axis=0)

target_cols = list(df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values)
lag_cols = target_cols+['age', 'segmento']
target_cols_diff = [i+'_diff' for i in target_cols]

print('creating lag features')
for lag in range(1, 6):
    lag_tmp = [i+str(lag) for i in lag_cols]
    df[lag_tmp] = df[lag_cols]
    df[lag_tmp] = df[lag_tmp].shift(lag).fillna(-2)
        
print('find out added products')
df[target_cols_diff] = df[target_cols]
df[target_cols_diff] = df[target_cols_diff].diff().fillna(0)
if df.fecha_dato.min()=='2015-06-28':
    df[df.fecha_dato=='2015-06-28', target_cols_diff] = \
        df[df.fecha_dato=='2015-06-28', target_cols]
        
print('melting')
dg = df.drop(target_cols, axis=1)

del df
del dt
gc.collect()

dg = dg.loc[(dg.fecha_dato=='2015-06-28')|(dg.fecha_dato=='2016-06-28')]
dg = pd.melt(dg, id_vars=[col for col in dg.columns 
                          if col not in target_cols_diff], 
                          value_vars=target_cols_diff)

print('extracting train and test data')
dg_train = dg.loc[(dg.value==1)&(dg.fecha_dato=='2015-06-28'),:].copy()
dg_test = dg.loc[dg.fecha_dato=='2016-06-28',:].copy()
dg_test.drop_duplicates('ncodpers', inplace=True)

del dg
gc.collect()

save_data('draft_hebbe2.pkl', (dg_train, dg_test))

#%%
dg_train, dg_test = read_data('draft_hebbe2.pkl')
y_train = dg_train.variable
dg_train.drop(['variable', 'value'], axis=1, inplace=True)
x_train = dg_train
dg_test.drop(['variable', 'value'], axis=1, inplace=True)
x_test = dg_test
x_train.drop(['fecha_dato'], axis=1, inplace=True)
x_test.drop(['fecha_dato'], axis=1, inplace=True)

save_data('data_for_train.pkl', (x_train, y_train, x_test))
