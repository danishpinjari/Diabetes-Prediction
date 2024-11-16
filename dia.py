import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fastapi.responses import HTMLResponse 

app = FastAPI()

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

class Diabetes(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float 
    DiabetesPedigreeFunction: float
    Age: int

#serve index.html
@app.get('/', response_class=HTMLResponse)
def index():
    # Open and read the index.html file and return its content as HTML
    with open('index.html', 'r', encoding='utf-8') as file:
        return file.read()

@app.post('/predict')
def predict(data: Diabetes):
    data = data.dict()
    Pregnancies = data['Pregnancies']
    Glucose = data['Glucose']
    BloodPressure = data['BloodPressure']
    SkinThickness = data['SkinThickness']
    Insulin = data['Insulin']   
    BMI = data['BMI']
    DiabetesPedigreeFunction = data['DiabetesPedigreeFunction']
    Age = data['Age']
    
    # Making a prediction
    prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Convert numpy type to a native Python type for JSON serialization
    result = int(prediction[0]) if isinstance(prediction[0], (np.integer, np.int64)) else prediction[0]

    return {'prediction': result}
