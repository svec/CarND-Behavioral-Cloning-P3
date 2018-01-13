#!/usr/bin/env python3

import csv
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, core

def read_csv(dirname):
    lines = []
    with open(dirname + "/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []

    for line in lines:

        # csv fields are:
        #   center_image left_image right_image steering_angle throttle break speed
        base_filename = os.path.basename(line[0])
        rel_path_image_filename = dirname + "/IMG/" + base_filename
        if rel_path_image_filename == None:
            print("ERROR: path does not exist:", rel_path_image_filename)
            sys.exit(1)
        # cv2.imread() reads in as BGR by default. Convert to RGB for our own sanity.
        image = cv2.imread(rel_path_image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

X_train, y_train = read_csv("car-data/run-0")
#X_train, y_train = read_csv("car-data/run-1")

print("X_train, y_train shapes:", X_train.shape, y_train.shape)

image_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
print("image shape:", image_shape)

model = Sequential()
model.add(core.Lambda(lambda x: x/255.0 - 0.5, input_shape=image_shape))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
