# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:00:01 2016

@author: celin
"""

import pandas as pd
import numpy as np
import gc

def process_chunk_data(df):
    """Clean chunk data
    """
    # date
    df.fecha_dato = pd.to_datetime(df.fecha_dato, format='%Y-%m-%d') 
    df.fecha_alta = pd.to_datetime(df.fecha_alta, format='%Y-%m-%d') 
    df['month'] = pd.DatetimeIndex(df.fecha_dato).month
    df['age'] = pd.to_numeric(df.age, errors='coerce')
    
    # ind_nuevo
    df.loc[df.ind_nuevo.isnull(), 'ind_nuevo'] = 1
           
    # antiguedad, seniority (in months)
    df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")
    df.loc[df.antiguedad.isnull(), "antiguedad"] = 0
    df.antiguedad = df.antiguedad.astype(int)
    df.loc[df.antiguedad<0, "antiguedad"] = 0

    # indrel, 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
    df.loc[df.indrel.isnull(),"indrel"] = 1
           
    # tipodom, nomprov, cod_prov
    df.drop(["tipodom","nomprov"],axis=1,inplace=True)
    df.loc[df.cod_prov.isnull(), 'cod_prov'] = -1
           
    return df
    
def find_may_june(df):
    """Find row indexes whose fecha_dato is in May or June
    """
    a1 = df.fecha_dato==pd.Timestamp('2015-05-28 00:00:00')
    a2 = df.fecha_dato==pd.Timestamp('2015-06-28 00:00:00')
    a3 = df.fecha_dato==pd.Timestamp('2016-05-28 00:00:00')
    a4 = df.fecha_dato==pd.Timestamp('2016-06-28 00:00:00')
    month_idx = a1.values | a2.values | a3.values | a4.values
    month_idx = np.where(month_idx==True)[0]
    
    return month_idx
    
def status_change(x):
    diffs = x.diff().fillna(0)# first occurrence will be considered Maintained, 
    #which is a little lazy. A better way would be to check if 
    #the earliest date was the same as the earliest we have in the dataset
    #and consider those separately. Entries with earliest dates later than that have 
    #joined and should be labeled as "Added"
    label = ["Added" if i==1 \
         else "Dropped" if i==-1 \
         else "Maintained" for i in diffs]
    return label

    
def post_clean_feature(df):
    """Clean data after loading all interesting rows, because the cleaning needs information from all the rows
    """
    # age
    df.loc[df.age<18, 'age'] = df.loc[(df.age>18) & (df.age<30), 'age'].mean(skipna=True)
    df.loc[df.age>100, 'age'] = df.loc[(df.age<100) & (df.age>30), 'age'].mean(skipna=True)
    df.age.fillna(df.age.mean(), inplace=True)
    df.age = df.age.astype(int)
    
    # fecha_alta
    dates = df.loc[:,"fecha_alta"].sort_values().reset_index()
    median_date = int(np.median(dates.index.values))
    df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]
           
    # renta
    df.renta = pd.to_numeric(df.renta, errors='coerce')
    incomes = df.loc[df.renta.notnull(),:].groupby("cod_prov").agg({"renta":{"MedianIncome":np.median}})
    incomes.sort_values(by=("renta","MedianIncome"),inplace=True)
    incomes.reset_index(inplace=True)
    incomes.cod_prov = incomes.cod_prov.astype("category", categories=[i for i in df.cod_prov.unique()],ordered=False)
    incomes.head()
    
    grouped = df.groupby("cod_prov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
    new_incomes = pd.merge(df,grouped,how="inner",on="cod_prov").loc[:, ["cod_prov","renta_y"]]
    new_incomes = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("cod_prov")
    df.sort_values("cod_prov",inplace=True)
    df = df.reset_index()
    new_incomes = new_incomes.reset_index()
    
    df.loc[df.renta.isnull(),"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()
    df.loc[df.renta.isnull(),"renta"] = df.loc[df.renta.notnull(),"renta"].median()
    df.sort_values(by="fecha_dato",inplace=True)
    
    return df
    
def post_clean_target(df):
    """Clean target
    """
    # ind_nomina_ult1
    df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
    df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0
           
    # 
    string_data = df.select_dtypes(include=["object"])
    missing_columns = [col for col in string_data if string_data[col].isnull().any()]
    
    df.loc[df.indfall.isnull(),"indfall"] = "N"
    df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
    df.tiprel_1mes = df.tiprel_1mes.astype("category")
    
    # As suggested by @StephenSmith
    map_dict = { 1.0  : "1",
                "1.0" : "1",
                "1"   : "1",
                "3.0" : "3",
                "P"   : "P",
                3.0   : "3",
                2.0   : "2",
                "3"   : "3",
                "2.0" : "2",
                "4.0" : "4",
                "4"   : "4",
                "2"   : "2"}
    
    df.indrel_1mes.fillna("P",inplace=True)
    df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
    df.indrel_1mes = df.indrel_1mes.astype("category")

    unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
    for col in unknown_cols:
        df.loc[df[col].isnull(),col] = "UNKNOWN"
        
    target_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
    for col in target_cols:
        df[col] = df[col].astype(int)
        
    unique_months = pd.DataFrame(pd.Series(df.fecha_dato.unique()).\
                                 sort_values()).reset_index(drop=True)
    unique_months["month_id"] = pd.Series(range(1,1+unique_months.size)) 
    unique_months["month_next_id"] = 1 + unique_months["month_id"]
    unique_months.rename(columns={0:"fecha_dato"},inplace=True)
    df = pd.merge(df,unique_months,on="fecha_dato")

    print('changing status')
    
    df.loc[:, target_cols] = df.loc[:, [i for i in target_cols]+["ncodpers"]].\
        groupby("ncodpers").transform(lambda x: x.diff().fillna(0))
    
    print('melting')    
    
    df = pd.melt(df, id_vars   = [col for col in df.columns if col not in target_cols],
                value_vars= [col for col in target_cols])
#    df = df.loc[df.value==1,:]
        
    return df
    
if __name__=='__main__':
    chunksize = 100000
    train_acc = []
    train = pd.read_csv('train_ver2.csv.zip', chunksize=chunksize,
                        dtype={"sexo":str, 
                               "ind_nuevo":str, 
                               "ult_fec_cli_1t":str,
                               "indext":str}, low_memory=False)
    for i, train_chunk in enumerate(train):
        train_chunk = process_chunk_data(train_chunk)
        month_idx_tmp = find_may_june(train_chunk)
        train_tmp = train_chunk.iloc[month_idx_tmp, :]
        if train_tmp.shape[0]>0:
            train_acc.append(train_tmp)
        print('Chunk {}, train_tmp shape {}'.format(i, train_tmp.shape))
        if i>25:
            break
        
    train_acc = pd.concat(train_acc)
    train_acc = post_clean_feature(train_acc)
    train_acc = post_clean_target(train_acc)
#    train_acc.to_csv('train_mj1.csv', index=False)
#    
#    test_acc = pd.read_csv('test_ver2.csv.zip', 
#                           dtype={"sexo":str, 
#                           "ind_nuevo":str, 
#                           "ult_fec_cli_1t":str,
#                           "indext":str}, low_memory=False)
#    test_acc = process_chunk_data(test_acc)
#    test_acc = post_clean_feature(test_acc)
