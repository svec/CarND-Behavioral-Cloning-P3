#!/usr/bin/env python3

import csv
import os, sys, copy
import cv2
import sklearn
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, core, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from random import shuffle

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dirs', 'car-data/run-0', "directories of car data")
flags.DEFINE_string('model_file_output', 'model.h5', "output model file name")
flags.DEFINE_integer('epochs', 5, "# epochs")
flags.DEFINE_integer('batch_size', 32, "batch_size")
flags.DEFINE_float('dropout_rate', 0.5, "dropout rate = % to drop")
flags.DEFINE_float('learning_rate', 0.001, "learning rate")
flags.DEFINE_string('model_type', 'nvidia', "NN model name")
flags.DEFINE_boolean('dont_run', False, "set dont_run=true to skip the final model training run")
flags.DEFINE_boolean('augment', True, "augment all data with horizontally flipping the image")
flags.DEFINE_boolean('refine', False, "Refine with new data and existing model, requires --model_file_input flag")
flags.DEFINE_string('model_file_input', None, "input model file name, only used with --refine")


CENTER_IMAGE_INDEX = 0
STEERING_ANGLE_INDEX = 3
FLIP_INDEX = 7

def fixup_paths(line, dirname):
    for ii in range(3): # uses CENTER_IMAGE_INDEX implicitly
        base_filename = os.path.basename(line[ii])
        rel_path_image_filename = dirname + "/IMG/" + base_filename
        if not os.path.exists(rel_path_image_filename):
            print("Skipping missing file:", rel_path_image_filename)
            return None
        line[ii] = rel_path_image_filename
    return line

def read_csv(dirname):
    print("Reading directory:", dirname)
    lines = []
    with open(dirname + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line = fixup_paths(line, dirname)
            if line == None:
                continue

            # Adding a 0 or 1 to determine if the image should be flipped
            # when it's processed.
            line.append(0)
            # Since we're changing a list, make a copy of the current list
            # before we change it below.
            # copy() works, and deepcopy() isn't required, because we changed
            # an int, for which copy() does the same thing as deepcopy().
            lines.append(copy.copy(line))
            line[FLIP_INDEX] = 1 # flip
            lines.append(line)
    return lines
    

total_images_returned = 0
total_generator_calls = 0

def generator(gen_name, samples, batch_size=32):
    global total_images_returned
    global total_generator_calls

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        #print(gen_name, "generator range(0,", num_samples, ",", batch_size, ")")
        for offset in range(0, num_samples, batch_size):
            #print(gen_name, "generator offset:", offset)
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[CENTER_IMAGE_INDEX]
                center_image = cv2.imread(name)
                # cv2.imread() reads in as BGR by default. Convert to RGB
                # for training the model.
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[STEERING_ANGLE_INDEX])
                should_flip = int(batch_sample[FLIP_INDEX])

                if should_flip == 1:
                    center_image = cv2.flip(center_image,1)
                    center_angle = center_angle * -1.0

                images.append(center_image)
                angles.append(center_angle)

                total_images_returned = total_images_returned + 1

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            total_generator_calls = total_generator_calls + 1
            #print(gen_name, "returning y_train shape:", y_train.shape)
            yield (X_train, y_train)

def main(_):
    data_dirs_string = FLAGS.data_dirs
    data_dirs_list = [x.strip() for x in data_dirs_string.split(',')]

    # r-0: fwd lap
    # r-1: fwd lap (very clean!) - horrible data though? Do not use!
    # r-2: fwd lap - problem data?
    # r-rev-0: reverse lap
    # r-swerve-0: swerving in from the right edges for 1 lap
    # r-swerve-1: swerving in from the left edges for 1 lap
    # r-swerve-2: swerving in from the right edges near the starting line
    # r-swerve-3: swerving in from the right edges near the starting line, and on+after the bridge
    # r-swerve-4: swerving in from the right edges in the post-bridge turn
    # r-curve-0: curve after bridge through the end

    samples = []

    for data_dir in data_dirs_list:
        lines = read_csv(data_dir)
        print("Added", len(lines), "samples from", data_dir)
        samples.extend(lines)

    print("Total samples:", len(samples), "(including horiz flipping)")
    #X_train = np.array(all_images)
    #y_train = np.array(all_measurements)
    #print("X_train, y_train shapes:", X_train.shape, y_train.shape)

    #image_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    #print("image shape:", image_shape) # should be: image shape: (160, 320, 3)
    image_shape = (160, 320, 3)

    model_type = FLAGS.model_type
    dropout_rate = FLAGS.dropout_rate
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    refine = FLAGS.refine
    model_file_input = FLAGS.model_file_input

    if refine:
        print("Refining existing model using model_file_input:", model_file_input)
        if not os.path.exists(model_file_input):
            print("ERROR: model_file_input doesn't exist:", model_file_input)
            sys.exit(1)
        model = load_model(model_file_input)
    else:
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
        print("Using batch size:", batch_size)
        
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

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print("train and validation sample count:", len(train_samples), len(validation_samples))

    train_generator = generator("train_generator", train_samples, batch_size=batch_size)
    validation_generator = generator("validation_generator", validation_samples, batch_size=batch_size)

    #model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs)
    model.fit_generator(train_generator, 
                        samples_per_epoch=len(train_samples), 
                        validation_data=validation_generator, 
                        nb_val_samples=len(validation_samples), 
                        nb_epoch=FLAGS.epochs,
                        verbose=1)
    
    model.save(FLAGS.model_file_output)
    print("Saved model in", FLAGS.model_file_output)
    print("total_images_returned by generator:", total_images_returned)
    print("total_generator_calls:", total_generator_calls)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
