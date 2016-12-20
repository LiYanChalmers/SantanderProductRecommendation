# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:00:07 2016

@author: celin
Add product history sum features
logloss->Mean:  0.506185338701  std:  0.00753883543626
LB->
"""


from santanderStart import *

train, _ = read_data('draft_hebbe4_train.pkl')
test_may, test_june = read_data('draft_hebbe4_test.pkl')

target_cols = train.variable.unique().tolist()
le = preprocessing.LabelEncoder()
le.fit(target_cols)

y_train = train.variable
y_train = le.transform(y_train)
y_train = pd.Series(y_train)

x_train = train.copy()
x_train.drop(['variable'], axis=1, inplace=True)
x_train.fillna(-1, inplace=True)

x_test = test_june.copy()

cols_drop = [col for col in test_may.columns if col not in target_cols]
cols_drop.remove('ncodpers')
test_may.drop(cols_drop, axis=1, inplace=True)

del train, test_june
gc.collect()

#%%
clfxgb = xgb.XGBClassifier(objective='multi:softprob', learning_rate=0.05, max_depth=8,
                           min_child_weight=1, subsample=0.7, colsample_bytree=0.7, seed=0, 
                           n_estimators=1000)

#%%
y_test_pred, y_train_pred, mlogloss, ntree = \
    cv_predict_xgb(clfxgb, x_train, y_train, x_test, cv=3, 
                   random_state=0, esr=5)

save_submission('sub1.csv', y_test_pred, le, x_test, test_may)

#%%
#y_test_pred_mean, y_train_pred_mean, y_test_pred, y_train_pred = \
#    cv_predict_xgb_repeat(clfxgb, x_train, y_train, x_test, 
#                          cv=3, random_state=0, rep=5, esr=10)
#save_data('lag4_result.pkl', (y_test_pred_mean, y_train_pred_mean, 
#                              y_test_pred, y_train_pred))


