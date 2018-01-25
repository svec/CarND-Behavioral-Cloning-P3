#!/usr/bin/env python3

import random
random.seed(42)
import numpy as np
np.random.seed(42)

import csv
import os, sys
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, core, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dirs', 'car-data/run-0', "directories of car data")
flags.DEFINE_string('model_file_output', 'model.h5', "output model file name")
flags.DEFINE_integer('epochs', 5, "# epochs")
flags.DEFINE_float('dropout_rate', 0.5, "dropout rate = % to drop")
flags.DEFINE_float('learning_rate', 0.001, "learning rate")
flags.DEFINE_string('model_type', 'nvidia', "NN model name")
flags.DEFINE_boolean('dont_run', False, "set dont_run=true to skip the final model training run")
flags.DEFINE_boolean('augment', True, "augment all data with horizontally flipping the image")


def read_csv(dirname, augment_with_horiz_flip=False):
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

        if augment_with_horiz_flip:
            images.append(cv2.flip(image,1))
            measurements.append(measurement * -1.0)

    return images, measurements

def main(_):
    data_dirs_string = FLAGS.data_dirs
    data_dirs_list = [x.strip() for x in data_dirs_string.split(',')]

    # r-0: fwd lap
    # r-1: fwd lap (very clean!) - horrible data though? Do not use!
    # r-2: fwd lap - problem data?
    # r-rev-0: reverse lap
    # r-swerve-0: swerving in from the right edges for 1 lap
    # r-swerve-1: swerving in from the left edges for 1 lap

    all_images = []
    all_measurements = []

    for data_dir in data_dirs_list:
        images, measurements = read_csv(data_dir, FLAGS.augment)
        all_images.extend(images)
        all_measurements.extend(measurements)
        print("len all_images, all_measurements:", len(all_images), len(all_measurements))

    X_train = np.array(all_images)
    y_train = np.array(all_measurements)
    print("X_train, y_train shapes:", X_train.shape, y_train.shape)

    image_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    print("image shape:", image_shape) # should be: image shape: (160, 320, 3)

    #model_type = "basic"
    #model_type = "lenet"
    model_type = FLAGS.model_type
    dropout_rate = FLAGS.dropout_rate
    learning_rate = FLAGS.learning_rate

    model = Sequential()

    # Poor man's normalization.
    # Image data starts at 0-255 (for RGB or YUV).
    # /255 makes it 0-1, then subtracting 0.5 centers it at 0.
    model.add(core.Lambda(lambda x: x/255.0 - 0.5, input_shape=image_shape))
    # Crops (removes) these pixels:
    # - top 65 pixels
    # - bottom 25 pixels
    # - 0 from the left or right
    model.add(Cropping2D(cropping=((65,25), (1,1))))

    print("Using model_type:", model_type)
    print("Using dropout rate:", dropout_rate)
    print("Using learning rate", learning_rate)

    if model_type == "basic":
    
        model.add(Flatten())
        model.add(Dense(1))
        # ____________________________________________________________________________________________________
        # Layer (type)                     Output Shape          Param #     Connected to
        # ====================================================================================================
        # lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
        # ____________________________________________________________________________________________________
        # cropping2d_1 (Cropping2D)        (None, 60, 318, 3)    0           lambda_1[0][0]
        # ____________________________________________________________________________________________________
        # flatten_1 (Flatten)              (None, 57240)         0           cropping2d_1[0][0]
        # ____________________________________________________________________________________________________
        # dense_1 (Dense)                  (None, 1)             57241       flatten_1[0][0]
        # ====================================================================================================
        # Total params: 57,241
        # Trainable params: 57,241
        # Non-trainable params: 0
    
    elif model_type == "lenet":
        model.add(Convolution2D(6, 5, 5, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Convolution2D(6, 5, 5, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))
        # X_train, y_train shapes: (1501, 160, 320, 3) (1501,)
        # image shape: (160, 320, 3)
        # Using model_type: lenet
        # ____________________________________________________________________________________________________
        # Layer (type)                     Output Shape          Param #     Connected to
        # ====================================================================================================
        # lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
        # ____________________________________________________________________________________________________
        # cropping2d_1 (Cropping2D)        (None, 60, 318, 3)    0           lambda_1[0][0]
        # ____________________________________________________________________________________________________
        # convolution2d_1 (Convolution2D)  (None, 56, 314, 6)    456         cropping2d_1[0][0]
        # ____________________________________________________________________________________________________
        # maxpooling2d_1 (MaxPooling2D)    (None, 28, 157, 6)    0           convolution2d_1[0][0]
        # ____________________________________________________________________________________________________
        # convolution2d_2 (Convolution2D)  (None, 24, 153, 6)    906         maxpooling2d_1[0][0]
        # ____________________________________________________________________________________________________
        # maxpooling2d_2 (MaxPooling2D)    (None, 12, 76, 6)     0           convolution2d_2[0][0]
        # ____________________________________________________________________________________________________
        # flatten_1 (Flatten)              (None, 5472)          0           maxpooling2d_2[0][0]
        # ____________________________________________________________________________________________________
        # dense_1 (Dense)                  (None, 120)           656760      flatten_1[0][0]
        # ____________________________________________________________________________________________________
        # dense_2 (Dense)                  (None, 84)            10164       dense_1[0][0]
        # ____________________________________________________________________________________________________
        # dense_3 (Dense)                  (None, 1)             85          dense_2[0][0]
        # ====================================================================================================
        # Total params: 668,371
        # Trainable params: 668,371
        # Non-trainable params: 0

    elif model_type == "nvidia":

        model.add(Convolution2D(24, 5, 5, border_mode="valid", subsample=(2,2), activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Convolution2D(36, 5, 5, border_mode="valid", subsample=(2,2), activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))

    else:
        print("ERROR: no valid model specified, model_type:",model_type)
        sys.exit(1)

    #model.summary()

    adam = optimizers.Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)

    if FLAGS.dont_run:
        print("NOT TRAINING")
        sys.exit(1)

    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs)
    
    model.save(FLAGS.model_file_output)
    print("Saved model in", FLAGS.model_file_output)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
