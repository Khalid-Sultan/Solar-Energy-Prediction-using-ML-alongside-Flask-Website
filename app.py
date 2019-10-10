from flask import Flask, render_template
from flask import request
from flask import json
from flask import jsonify

import requests
import time
import datetime
import json

import numpy as np
import pandas as pd

from keras.models import load_model

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=["POST"])
def generate():
    print("Loading Keras Model")
    model = get_model()
    variables = request.get_json(force=True)
    cloudCoverage = variables['cloudCoverage']
    visibility = variables['visibility']
    relativeHumidity = variables['relativeHumidity']
    windSpeed = variables['windSpeed']
    temperature = variables['temperature']
    dewPoint = variables['dewPoint']
    stationPressure = variables['stationPressure']
    altimeter = variables['altimeter']
    given_values = {
        'Cloud Coverage': [cloudCoverage],
        'Visibility': [visibility],
        'Temperature': [relativeHumidity],
        'Dew Point': [windSpeed],
        'Relative Humidity': [temperature],
        'Wind Speed': [dewPoint],
        'Station Pressure': [stationPressure],
        'Altimeter': [altimeter]
    }
    clean_values = pd.DataFrame(given_values)
    clean_values = clean_values.astype('float32')
    prediction = model.predict(clean_values)
    result = {
        'value': '{}'.format(prediction[0][0])
    }
    return jsonify(result)


@app.route("/generateTimeline")
def generate_timeline():
    print("Loading Keras Model")
    countries = get_countries()
    datas = []
    for i in countries.keys():
        weather = get_location_weather_now(countries[i][0], countries[i][1])
        keys = ['cloudCover', 'visibility', 'humidity', 'windSpeed',
                'temperature', 'dewPoint', 'pressure', 'pressure']
        variables = weather['currently']
        if not check_if_all_data_is_available(variables, keys):
            print('Skipped Country {} with Latitude {} and Longitude {}'.format(i, countries[i][0], countries[i][1]))
            continue
        cloudCoverage = variables['cloudCover']
        visibility = variables['visibility']
        relativeHumidity = variables['humidity']
        windSpeed = variables['windSpeed']
        temperature = variables['temperature']
        dewPoint = variables['dewPoint']
        stationPressure = variables['pressure']
        altimeter = variables['pressure']
        given_values = {
            'Cloud Coverage': [cloudCoverage],
            'Visibility': [visibility],
            'Temperature': [relativeHumidity],
            'Dew Point': [windSpeed],
            'Relative Humidity': [temperature],
            'Wind Speed': [dewPoint],
            'Station Pressure': [stationPressure],
            'Altimeter': [altimeter]
        }
        clean_values = pd.DataFrame(given_values)
        clean_values = clean_values.astype('float32')
        print("Loaded weather data for City {}".format(i))
        datas.append(clean_values)
    df = pd.concat(datas)
    results = {}
    model = get_model()
    predictions = model.predict(df)
    counter = 0
    print(predictions.shape)
    for i in predictions[0]:
        print('Predicted Solar Energy for Country {}/{}'.format(counter, predictions.shape[0]))
        counter += 1
        savant = 0
        key = ""
        for j in countries.keys():
            if counter == savant:
                key = j
                break
            savant += 1
        results[key] = '{}'.format(i)
    pred_df = pd.DataFrame(results, columns=['Country', 'Value'])
    pred_df = pred_df.astype('float32')
    export = pred_df.to_json(orient='split')
    return jsonify(export)


@app.route("/fillCurrentLocationWeather")
def get_current_location_weather_now():
    latitude, longitude = get_my_current_location()
    url = 'https://api.darksky.net/forecast/7e5553d96e98639d454789b49d62e88d/{},{}'.format(latitude, longitude)
    r = requests.get(url)
    j = json.loads(r.text)
    return jsonify(j)


def get_location_weather_now(latitude, longitude):
    url = 'https://api.darksky.net/forecast/7e5553d96e98639d454789b49d62e88d/{},{}'.format(latitude, longitude)
    r = requests.get(url)
    j = json.loads(r.text)
    return j


def get_my_current_location():
    url = 'http://api.ipstack.com/check?access_key=ec395806777e96f1e9b2e9c3430cad7b'.format(request.remote_addr)
    r = requests.get(url)
    j = json.loads(r.text)
    return j['latitude'], j['longitude']


def get_model():
    model = load_model('Solar_Energy_Prediction.h5')
    print("Solar Prediction Model has been loaded")
    return model


def get_countries():
    countries = {}
    with open('countries.json') as json_file:
        data = json.load(json_file)
        for p in data:
            countries[p['capital']] = p['latlng']
    return countries


def check_if_all_data_is_available(variables, keys):
    for key in keys:
        if key not in variables.keys():
            print("This data was not available {}".format(key))
            return False
    return True


if __name__ == '__main__':
    app.run()
