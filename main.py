from fastapi import FastAPI,File, Form, UploadFile
from pydantic import BaseModel
import pandas as pd
from io import StringIO
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier


app=FastAPI()


class IrisSpecies(BaseModel):
    Searches: float
    Filters: float
    Shortlist: float
    Enquiries: float
    Photoclicks: float
    PageView: float
        
                                        
@app.get('/')
def index():
    
    return {'message': "This is the home page of this API. Go to /apiv1/ or /apiv2/?name="}
df1 = pd.read_csv('dfl.csv')
df1 = df1[["Searches","Filters","Shortlist","Enquiries","Photoclicks","PageView"]]


@app.post('/predict')
async def predict_species(result: IrisSpecies):
    data12 = result.dict()
    data=pd.DataFrame([data12])
    print(data)
    
    data_1=(data-df1.min())/(df1.max()-df1.min())
    print(data_1)
    
    
    loaded_model = pickle.load(open('fmodel.pkl', 'rb'))
    #data_in = [[data_1['Searches'], data_1['Filters'], data_1['Shortlist'], data_1['Enquiries'], data_1['Photoclicks'], data_1['PageView']]]
    prediction = loaded_model.predict(data_1)
    #probability = loaded_model.predict_proba(data_in).max()
    
    return {
    'prediction': prediction[0],
    # 'probability': probability
    }


