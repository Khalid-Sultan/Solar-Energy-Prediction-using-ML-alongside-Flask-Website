# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 07:38:44 2019
Updated on Wen Oct  10 01:25:07 2019
@author: gemme
"""

import os
import numpy as np
from keras.models import Sequential
from keras import layers
import pandas as pd


# Set y values of data to lie between 0 and 1
def normalize_data(dataset, data_min, data_max):
    data_std = (dataset - data_min) / (data_max - data_min)
    test_scaled = data_std * (np.amax(data_std) - np.amin(data_std)) + np.amin(data_std)
    return test_scaled


def import_data(train_dataframe, dev_dataframe, test_dataframe):
    dataset = train_dataframe.values
    dataset = dataset.astype('float32')
    # Include all 8 initial factors (Cloud Coverage ; Visibility ; Temperature ; Dew Point ; Relative Humidity ; Wind Speed ; Station Pressure ; Altimeter)
    max_test = np.max(dataset[:, 8])
    min_test = np.min(dataset[:, 8])
    scale_factor = max_test - min_test
    max = np.empty(13)
    min = np.empty(13)

    # Create training dataset
    for i in range(0, 10):
        min[i] = np.amin(dataset[:, i], axis=0)
        max[i] = np.amax(dataset[:, i], axis=0)
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    train_data = dataset[:, 0:8]
    train_labels = dataset[:, 8]

    # Create dev dataset
    dataset = dev_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, 10):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    dev_data = dataset[:, :8]
    dev_labels = dataset[:, 8]

    # Create test dataset
    dataset = test_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, 10):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    test_data = dataset[:, 0:8]
    test_labels = dataset[:, 8]

    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor


def build_model(train_data):
    model = Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def mse(predicted, observed):
    return np.sum(np.multiply((predicted - observed), (predicted - observed))) / predicted.shape[0]


def main():
    train_dataframe = pd.read_csv('weather_train.csv', sep=";", engine='python', header=None)
    dev_dataframe = pd.read_csv('weather_dev.csv', sep=";", engine='python', header=None)
    test_dataframe = pd.read_csv('weather_test.csv', sep=";", engine='python', header=None)

    #Drop first column
    train_dataframe.drop(train_dataframe.columns[0], axis=1)
    dev_dataframe.drop(dev_dataframe.columns[0], axis=1)
    test_dataframe.drop(test_dataframe.columns[0], axis=1)

    print(train_dataframe.head(1))
    print(dev_dataframe.head(1))
    print(test_dataframe.head(1))

    train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor = import_data(train_dataframe,
                                                                                                       dev_dataframe,
                                                                                                       test_dataframe)
    time_steps = 1
    X_train = np.reshape(train_data, (train_data.shape[0] // time_steps, time_steps, train_data.shape[1]))
    X_dev = np.reshape(dev_data, (dev_data.shape[0] // time_steps, time_steps, dev_data.shape[1]))
    X_test = np.reshape(test_data, (test_data.shape[0] // time_steps, time_steps, test_data.shape[1]))
    Y_train = np.reshape(train_labels, (train_labels.shape[0] // time_steps, time_steps, 1))
    Y_dev = np.reshape(dev_labels, (dev_labels.shape[0] // time_steps, time_steps, 1))
    Y_test = np.reshape(test_labels, (test_labels.shape[0] // time_steps, time_steps, 1))

    model = build_model(train_data)
    history = model.fit(train_data, train_labels, epochs=500, batch_size=40, validation_split=0.2)
    test_mse_score, test_mae_score = model.evaluate(dev_data, dev_labels, verbose=0)
    print('Average prediction you are off by is ', test_mae_score)
    # save model and architecture to single file
    model.save("Solar_Energy_Prediction.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    main()
