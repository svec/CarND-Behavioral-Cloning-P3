#!/usr/bin/env python3

import csv
import os, sys
import random
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, core, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

random.seed(42)
np.random.seed(42)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dirs', 'car-data/run-0', "directories of car data")
flags.DEFINE_string('model_file_output', 'model.h5', "output model file name")
flags.DEFINE_integer('epochs', 1, "# epochs")


def read_csv(dirname):
    print("Reading directory:", dirname)
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
        if not os.path.exists(rel_path_image_filename):
            print("Skipping missing file:", rel_path_image_filename)
            continue
        # cv2.imread() reads in as BGR by default. Convert to RGB for our own sanity.
        image = cv2.imread(rel_path_image_filename)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    return images, measurements

def main(_):
    data_dirs_string = FLAGS.data_dirs
    data_dirs_list = [x.strip() for x in data_dirs_string.split(',')]

    all_images = []
    all_measurements = []

    for data_dir in data_dirs_list:
        images, measurements = read_csv(data_dir)
        all_images.extend(images)
        all_measurements.extend(measurements)
        print("len all_images, all_measurements:", len(all_images), len(all_measurements))

    X_train = np.array(all_images)
    y_train = np.array(all_measurements)
    print("X_train, y_train shapes:", X_train.shape, y_train.shape)

    image_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    print("image shape:", image_shape) # should be: image shape: (160, 320, 3)

    model = Sequential()
    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(1))
    #model.summary()

    model.compile(loss='mse', optimizer='adam')

    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs)
    
    model.save(FLAGS.model_file_output)
    print("Saved model in", FLAGS.model_file_output)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
