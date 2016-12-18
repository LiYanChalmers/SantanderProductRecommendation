#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:01:58 2016

@author: lyaa
"""

from santanderStart import *

##%%
#df = pd.read_csv('train_ver2.csv.zip', low_memory=False)
#
#dt = pd.read_csv('test_ver2.csv.zip', low_memory=False)
#target_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
#dummy = np.zeros((dt.shape[0], 24))
#dummy = pd.DataFrame(dummy, columns=target_cols, index=dt.index)
#dt = dt.join(dummy)
#
#ntrain = df.shape[0]
#ntest = dt.shape[0]
#
#df = pd.concat([df, dt])
#
##%%
#df = process_chunk_data(df)
#df = post_clean_feature(df)
#
##%%
#df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
#df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0
#
#string_data = df.select_dtypes(include=["object"])
#missing_columns = [col for col in string_data if string_data[col].isnull().any()]
#
#df.loc[df.indfall.isnull(),"indfall"] = "N"
#df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
#df.tiprel_1mes = df.tiprel_1mes.astype("category")
#
## As suggested by @StephenSmith
#map_dict = { 1.0  : "1",
#            "1.0" : "1",
#            "1"   : "1",
#            "3.0" : "3",
#            "P"   : "P",
#            3.0   : "3",
#            2.0   : "2",
#            "3"   : "3",
#            "2.0" : "2",
#            "4.0" : "4",
#            "4"   : "4",
#            "2"   : "2"}
#
#df.indrel_1mes.fillna("P",inplace=True)
#df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
#df.indrel_1mes = df.indrel_1mes.astype("category")
#
#unknown_cols = [col for col in missing_columns 
#                if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
#for col in unknown_cols:
#    df.loc[df[col].isnull(),col] = "UNKNOWN"
#
#target_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
#for col in target_cols:
#    df[col] = df[col].astype(int)
#    
#unique_months = pd.DataFrame(pd.Series(df.fecha_dato.unique()).\
#                             sort_values()).reset_index(drop=True)
#unique_months["month_id"] = pd.Series(range(1,1+unique_months.size)) 
#unique_months["month_next_id"] = 1 + unique_months["month_id"]
#unique_months.rename(columns={0:"fecha_dato"},inplace=True)
#df = pd.merge(df,unique_months,on="fecha_dato")
#
#del string_data
#gc.collect()
#
##%%
#df.drop(['index', 'month', 'month_id', 'month_next_id'], axis=1, inplace=True)
#df.to_csv('train_clean.csv')
#
##%%
#df = pd.read_csv('train_clean.csv')
#
##%%
#cols = df.columns.tolist()
#for c in cols:
#    print(c, '->', len(df[c].unique()))
#    
##%%
#df.drop(['Unnamed: 0'], axis=1, inplace=True)
#
##%% ind_empleado
#le = preprocessing.LabelEncoder()
#df.ind_empleado = le.fit_transform(df.ind_empleado)
#df.ind_empleado.unique()
#
##%% pais_residencia
#df.pais_residencia = le.fit_transform(df.pais_residencia)
#df.pais_residencia.unique()
#
##%% sexo 
#df.sexo = le.fit_transform(df.sexo)
#df.sexo.unique()
#
##%% fecha_alta
#df.fecha_alta = pd.to_datetime(df.fecha_alta)
#fecha_alta_min = df.fecha_alta.min()
#df.fecha_alta = (df.fecha_alta-fecha_alta_min)/np.timedelta64(1, 'D')
#
##%% ind_nuevo
#df.ind_nuevo = le.fit_transform(df.ind_nuevo)
#df.ind_nuevo.unique()
#
##%% ult_fec_cli_1t
#df.fecha_dato = pd.to_datetime(df.fecha_dato)
#df.ult_fec_cli_1t.replace(to_replace='UNKNOWN', value=0, inplace=True)
#df.ult_fec_cli_1t = pd.to_datetime(df.ult_fec_cli_1t, errors='ignore')
#df.ult_fec_cli_1t = (df.ult_fec_cli_1t-df.fecha_dato)/np.timedelta64(1, 'D')
#df.loc[df.ult_fec_cli_1t<-10000,'ult_fec_cli_1t'] = -999
#
##%% indrel_1mes
#replace_dict = {'indrel_1mes':{'1':1, '2':2, '3':3, '4.0':4, '4':4, 'P':5}}
#df.replace(to_replace=replace_dict, inplace=True)
#
##%% tiprel_1mes
#df.tiprel_1mes = le.fit_transform(df.tiprel_1mes)
#
##%% indresi
#df.indresi = le.fit_transform(df.indresi)
#
##%% indresi
#df.indext = le.fit_transform(df.indext)
#
##%% conyuemp
#df.conyuemp = le.fit_transform(df.conyuemp)
#
##%% canal_entrada
#df.canal_entrada = le.fit_transform(df.canal_entrada)
#
##%% indfall
#df.indfall = le.fit_transform(df.indfall)
#
##%% cod_prov
#df.cod_prov = le.fit_transform(df.cod_prov)
#
##%% ind_actividad_cliente
#df.ind_actividad_cliente.fillna(-1)
#
##%% segmento
#df.segmento = le.fit_transform(df.segmento)
#
##%% fecha_dato
##fecha_dato_min = df.fecha_dato.min()
##df.fecha_dato = (df.fecha_dato-fecha_dato_min)/np.timedelta64(1, 'D')
#
##%% 
#df_train = df.iloc[:ntrain,:]
#df_test = df.iloc[ntrain:,:]
#df_train.to_csv('train_clean.csv', index=False)
#df_test.to_csv('test_clean.csv', index=False)

#%%
df = pd.read_csv('train_clean.csv')
dt = pd.read_csv('test_clean.csv')
df = pd.concat([df, dt])

#%%
#ds = df.sort_values(by=['ncodpers', 'fecha_dato'], axis=0)

#%% 
ncodpers_list = df.ncodpers.unique()
#n_ncodpers = len(ncodpers_list)
#customer_record = {}
#customer_record_melt = {}
#customer_train = {}
#customer_test ={}
df_ncodpers = df.groupby('ncodpers')
mp_lag_in = (df_ncodpers, ncodpers_list)
mp_result = mp_lag(mp_lag_in)
save_data('mp_lag.pkl', (mp_result, ncodpers_list))
#for i, cust in enumerate(ncodpers_list):
#    customer_record[i], customer_record_melt[i], customer_train[i], \
#        customer_test[i] = add_lag_features(df_ncodpers.get_group(cust).copy())
#    if i%100==0:
#        print('{}/{}'.format(i, n_ncodpers))
        
#%%
#target_cols = list(df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values)
#lag_cols = target_cols+['age', 'segmento']
#target_cols_diff = [i+'_diff' for i in target_cols]
#a = ncodpers_dict[10]
#a[target_cols_diff] = a[target_cols]
#a[target_cols_diff] = a[target_cols_diff].diff().fillna(missing_values).fillna(0)
#if a.fecha_dato.min()=='2015-06-28':
#    a[a.fecha_dato=='2015-06-28', target_cols_diff] = a[a.fecha_dato=='2015-06-28', target_cols]

#%%


