import json
import time

import pandas as pd
import requests
from flask import Flask, render_template
from flask import json
from flask import jsonify
from flask import request
from keras.models import load_model

app = Flask(__name__)
api_key = "0113592eb98c0feeefac0af2898f1c5f"


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
        'value': '{}'.format(round(float(prediction[0][0]), 1))
    }
    return jsonify(result)


@app.route("/generateTimeline")
def generateTimeline():
    print("Loading Keras Model")
    weather = get_current_location_weather_week()
    keys = ['cloudCover', 'visibility', 'humidity', 'windSpeed',
            'temperature', 'dewPoint', 'pressure', 'pressure']
    daily = weather['daily']['data']
    datas = []
    datekeys = {}
    print(daily)
    for variables in daily:
        date = time.strftime('%Y-%m-%d', time.localtime(variables['time']))
        datekeys[date] = 1

        cloudCoverage = variables['cloudCover']
        visibility = variables['visibility']
        relativeHumidity = variables['humidity']
        windSpeed = variables['windSpeed']
        temperature = (variables['temperatureMax'] + variables['temperatureMin']) / 2
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
        datas.append(clean_values)
    df = pd.concat(datas)
    results = {}
    model = get_model()
    predictions = model.predict(df)
    counter = 0
    for i in predictions[0:]:
        savant = 0
        key = ""
        for j in datekeys.keys():
            if counter == savant:
                key = j
                break
            savant += 1
        counter += 1
        results[key] = '{}'.format(i[0])
    json_results = []
    counter = 0
    for i in results.keys():
        json_results_temp = {'date': i, 'value': float(results[i]), 'max': max(results.values()),
                             'min': min(results.values())}
        counter += 1
        json_results.append(json_results_temp)
    print(json_results)
    return jsonify(json_results)


@app.route("/compare")
def compare():
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
    for i in predictions[0:]:
        print('Predicted Solar Energy for Country {}/{}'.format(counter+1, predictions.shape[0]))
        savant = 0
        key = ""
        for j in countries.keys():
            if counter == savant:
                key = j
                break
            savant += 1
        counter += 1
        results[key] = '{}'.format(i[0])
    json_results = []
    for i in results.keys():
        json_results_temp = {'Country': i, 'Value': results[i], 'max': max(results.values()),
                             'min': min(results.values())}
        json_results.append(json_results_temp)
    return jsonify(json_results)


@app.route("/fillCurrentLocationWeather")
def get_current_location_weather_now():
    latitude, longitude = get_my_current_location()
    url = 'https://api.darksky.net/forecast/{}/{},{}'.format(api_key, latitude, longitude)
    # url = 'https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid='.format(latitude,longitude,api_key)
    r = requests.get(url)
    j = json.loads(r.text)
    return jsonify(j)


def get_current_location_weather_week():
    latitude, longitude = get_my_current_location()
    url = 'https://api.darksky.net/forecast/{}/{},{}'.format(api_key, latitude, longitude)
    r = requests.get(url)
    j = json.loads(r.text)
    return j


def get_location_weather_now(latitude, longitude):
    url = 'https://api.darksky.net/forecast/{}/{},{}'.format(api_key, latitude, longitude)
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
