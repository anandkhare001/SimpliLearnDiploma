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
    pickup_lattitude: float
    pickup_longitude: float
    dropoff_lattitude: float
    dropoff_longitude: float
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
@app.post('/predict')
def predict_fare(ride_details: UberFare):
    data = ride_details.model_dump()
    pickup_lattitude = data['pickup_lattitude']
    pickup_longitude = data['pickup_longitude']
    dropoff_lattitude = data['dropoff_lattitude']
    dropoff_longitude = data['dropoff_longitude']
    passenger_count = data['passenger_count']
    day = data['day']
    hour = data['hour']
    weekday = data['weekday']
    month = data['month']
    year = data['year']

    # Make prediction
    prediction = model.predict([[pickup_longitude, pickup_lattitude, dropoff_longitude, dropoff_lattitude, passenger_count, day, hour, weekday, month, year]])

    return {'Ride fare predicted': prediction}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)

