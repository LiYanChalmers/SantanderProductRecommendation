#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:01:58 2016

@author: lyaa
"""

from santanderStart import *

df = pd.read_csv('train_clean.csv')
dt = pd.read_csv('test_clean.csv')
df = pd.concat([df, dt])

ncodpers_list = df.ncodpers.unique()
df_ncodpers = df.groupby('ncodpers')
result = {}
for i, n in enumerate(ncodpers_list):
    result[i] = add_lag_features(df_ncodpers.get_group(n).copy())
    
save_data('results_lag.pkl', result)
