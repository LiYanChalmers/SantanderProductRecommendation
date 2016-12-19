#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 08:54:23 2016

@author: lyaa
"""

from santanderStart import *

x_train, y_train, x_test = read_data('data_for_train.pkl')
y_train.replace(y_train.unique().tolist(), target_cols, inplace=True)
x_train = x_train.as_matrix()
x_test = x_test.as_matrix()
le = preprocessing.LabelEncoder()
le.fit(target_cols)
y_train = le.transform(y_train)

#x_train, y_train = read_data('train_mj2.pkl')
#x_test = read_data('test_mj2.pkl')

#%%
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.05
param['max_depth'] = 8
param['silent'] = 1
param['num_class'] = 22
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 0
num_rounds = 5

#%%
y_test_pred, y_train_pred, mlogloss, ntree = \
    cv_predict_xgb(param, num_rounds, x_train, y_train, x_test, cv=2, 
                   random_state=0, esr=10)

#%%
save_submission('sub1.csv', y_test_pred, le)
