from fastapi import FastAPI,File, Form, UploadFile, Response
from fastapi import Request, FastAPI
from pydantic import BaseModel
import pandas as pd
from io import StringIO
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import json


app=FastAPI()

df1 = pd.read_csv('dfl.csv')
df1 = df1[["Searches","Filters","Shortlist","Enquiries","Photoclicks","PageView"]]
                                               
@app.get('/')
def index():
    
    return {'message': "This is the home page of this API. Go to /apiv1/ or /apiv2/?name="}
  
  
  
@app.post('/predict')
async def predict_species(request:Request):
    data12 = await request.json()
    initial_data = pd.DataFrame(data12)
    print(initial_data.head())

    test=initial_data[['Searches','Filters','Shortlist','Enquiries','Photoclicks','PageView']]
    data_1=(test-df1.min())/(df1.max()-df1.min())#Normaalized the input data
    print("data",data_1.head())

    loaded_model = pickle.load(open('model_L7d.pkl', 'rb'))#Load the model
    # prediction = loaded_model.predict(data_1)
    prediction = loaded_model.predict(data_1)
    prediction2 = loaded_model.predict_proba(data_1)

    initial_data["Result"] = prediction
    print(initial_data.head())
    initial_data[["Prob_0","Prob_1"]] = prediction2
    print(initial_data.head())
 
    orient="records"
    final_result=initial_data.to_json(orient=orient)
    return Response(final_result)

# Old code
# @app.post('/predict')
# async def predict_species(request:Request):
#     data12 = await request.json()
#     test = pd.DataFrame(data12)
#     data_1=(test-df1.min())/(df1.max()-df1.min())#Normaalized the input data

#     loaded_model = pickle.load(open('fmodel.pkl', 'rb'))#Load the model
#     prediction = loaded_model.predict(data_1)
#     data_1["Result"]=prediction#store final value in result

#     data1=[['Searches','Filters','Shortlist','Enquiries','Photoclicks','PageView','Result']]
#     orient="records"
#     result=data_1.to_json(orient=orient)
    
#     return Response(result)


