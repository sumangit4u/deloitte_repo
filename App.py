from flask import Flask, render_template, request, Response,session,redirect,jsonify, url_for
from flask.helpers import make_response
import requests
from flask import abort
import pandas as pd
import numpy as np
import json
from pickle import dump,load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,f1_score,auc
from utils import get_json_data,get_tsv_data,get_custom_data

#######################App related Stuff###################################
app = Flask(__name__)
################### This is how we do attribute prigraming in PYTHON #########
@app.route('/')
def train():
   try:
      tsv_df = get_tsv_data()
   except FileNotFoundError:
        resp = "Did not find tsv data while training."
        return resp   
   try:
      custom_df = get_custom_data()
   except FileNotFoundError:
        resp = "Did not find samples.custom while training."
        return resp
   try:        
      json_df = get_json_data()
   except FileNotFoundError:
        resp = "Did not find json data while training."
        return resp   

   final_df = custom_df.merge(json_df)
   final_df = final_df.merge(tsv_df) 

   feature_cols = [col for col in final_df.columns if 'feature' in col and col not in ['feature_9','feature_8']]
   X=final_df[feature_cols]
   Y=final_df['Target']

   x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,shuffle = True,random_state=42)

   le = LabelEncoder()
   le.fit(x_train['feature_2'])
   dump(le, open('model/le.pkl', 'wb'))
   x_train['feature_2'] = le.transform(x_train['feature_2'])
   x_test['feature_2'] = le.transform(x_test['feature_2'])

   sc = StandardScaler()
   sc.fit(x_train)
   dump(sc, open('model/sc.pkl', 'wb'))
   x_train_sc = sc.transform(x_train)
   x_test_sc=sc.transform(x_test)

   lr = LogisticRegression()
   lr.fit(x_train_sc,y_train)
   dump(lr, open('model/lr.pkl', 'wb'))
   pred = lr.predict(x_test_sc)
   pred_train = lr.predict(x_train_sc)
   print('Train F1 Score:',f1_score(y_train,pred_train))
   print('Test Accuracy:',f1_score(y_test,pred))  
   resp = 'Train F1 Score:' + str(f1_score(y_train,pred_train))+'\n'+'Test F1 Score:' + str(f1_score(y_test,pred))


   return resp

@app.route('/predict')
def predict():
   feature_1=float(request.args.get('feature_1'))
   feature_2=request.args.get('feature_2')
   feature_3=float(request.args.get('feature_3'))
   feature_4=float(request.args.get('feature_4'))
   feature_5=float(request.args.get('feature_5'))
   feature_6=float(request.args.get('feature_6'))
   feature_7=float(request.args.get('feature_7'))
   # feature_8=float(request.args.get('feature_8'))
   # feature_9=float(request.args.get('feature_9'))
   feature_10=float(request.args.get('feature_10'))

   le_load = load(open('model/le.pkl','rb'))
   feature_2 = le_load.transform([feature_2])[0]
   
   test_df = np.array([feature_10,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7])   
   test_df = test_df.reshape(1,-1)

   print(test_df.shape)
   sc_load = load(open('model/sc.pkl','rb'))
   test_df = sc_load.transform(test_df)
   print(test_df.shape)

   model = load(open('model/lr.pkl', 'rb'))
   pred = model.predict(test_df)
   print(pred)

   return 'Prediction is '+str(pred[0])
##########################################################
if __name__ == "__main__":
    app.run(debug=True)
##########################################################
