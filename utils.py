import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,f1_score,auc

def get_custom_data():
    try:
        with open('samples/samples.custom') as fh:
            data = fh.read()
    except FileNotFoundError:
        raise

    data = data.split('\n')[:-1]
    list_9 = []
    list_10 = []
    list_uid = []
    for rec in data:
        rec_list=rec.split('=')
        list_uid.append(rec_list[1].split('feature')[0])
        list_9.append(rec_list[2].split('feature')[0])
        list_10.append(rec_list[3])
        
    custom_df = pd.DataFrame(zip(list_9,list_10,list_uid),columns=['feature_9','feature_10','user_id'])  

    return custom_df

def get_tsv_data():
    try:
        tsv_df = pd.read_csv('samples/samples.tsv',sep='\t',header = None)
    except FileNotFoundError:
        raise
    tsv_df.rename({0:'user_id',1:'Target'},inplace=True,axis=1)  

    return tsv_df


def get_json_data():
    json_df_cols=[]
    try:
        with open('samples/samples.json') as fh:
            data = fh.read()
    except FileNotFoundError:
        raise
    data = data.split('\n')[0:-1]

    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    list6=[]
    list7=[]
    list8=[]
    list_id =[]

    for rec in data:
        for k,v in json.loads(rec).items():
            if k == 'feature_1':
                list1.append(v)
            if k == 'feature_2':
                list2.append(v)
            if k == 'feature_3':
                list3.append(v)
            if k == 'feature_4':
                list4.append(v)
            if k == 'feature_5':
                list5.append(v)
            if k == 'feature_6':
                list6.append(v)
            if k == 'feature_7':
                list7.append(v)
            if k == 'feature_8':
                list8.append(v)  
            if k == 'user_id':
                list_id.append(v) 
    json_df = pd.DataFrame(zip(list1,list2,list3,list4,list5,list6,list7,list8,list_id),
                            columns=['feature_1','feature_2','feature_3','feature_4','feature_5','feature_6',
                                    'feature_7','feature_8','user_id'])    

    return json_df   