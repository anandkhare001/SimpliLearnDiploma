from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd
import config
from data import load_pipeline


app = FastAPI()
model = load_pipeline(config.MODEL_NAME)


# Perform parsing of inputs features
class UberFare(BaseModel):
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: float
    day: float
    hour: float
    weekday: float
    month: float
    year: float


@app.get('/')
def index():
    return {'message': 'Welcome to Uber Fare Prediction App'}


# defining the function which will make the prediction using the data which the user inputs
@app.post('/prediction')
def predict_fare(ride_details: UberFare):
    inputData = ride_details.model_dump()

    # Make prediction
    prediction = model.predict([[inputData['pickup_longitude'],
                                 inputData['pickup_latitude'],
                                 inputData['dropoff_longitude'],
                                 inputData['dropoff_latitude'],
                                 inputData['passenger_count'],
                                 inputData['day'],
                                 inputData['hour'],
                                 inputData['weekday'],
                                 inputData['month'],
                                 inputData['year']]])

    return {'Ride fare predicted': str(prediction)}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)

