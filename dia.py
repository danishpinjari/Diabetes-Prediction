import pandas as pd
import pickle
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow cross-origin requests for development (you can restrict origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the input data model
class Diabetes(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Serve the index.html page
@app.get('/', response_class=HTMLResponse)
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

# Endpoint to make predictions
@app.post('/predict')
def predict(data: Diabetes):
    data_dict = data.dict()
    input_data = [
        data_dict['Pregnancies'],
        data_dict['Glucose'],
        data_dict['BloodPressure'],
        data_dict['SkinThickness'],
        data_dict['Insulin'],
        data_dict['BMI'],
        data_dict['DiabetesPedigreeFunction'],
        data_dict['Age']
    ]
    
    # Make the prediction
    prediction = model.predict([input_data])
    result = int(prediction[0]) if isinstance(prediction[0], (np.integer, np.int64)) else prediction[0]

    return {'prediction': result}
