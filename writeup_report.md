# **Behavioral Cloning** 

## Writeup for Chris Svec

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Convince my wife that I am *not* just playing a video game; I'm playing a video game *for science!*
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./writeup_images/center-1.jpg "center lane driving"
[image3]: ./writeup_images/recovery-1.jpg "Recovery Image 1"
[image4]: ./writeup_images/recovery-2.jpg "Recovery Image 2"
[image5]: ./writeup_images/recovery-3.jpg "Recovery Image 3"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results (you're reading it)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file (which is the original Udacity-provided drive.py file), the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the NVIDIA network from the paper, "End to End Learning for Self-Driving Cars."

All line numbers refer to the model.py file in github.

The initial image processing converts the BGR images to RGB (line 92), normalizes the pixel data (line 169), and removes the top and bottom of the image which contains useless data (line 174).

The rest of the model starts at `model_type == "nvidia"` in model.py line 243.

The NVIDIA network has five convolutional layers, each with RELU activation and a
dropout layer. The first three layers use 5x5 filters, followed by two
convolutional layers with 3x3 filters. (lines 247-256)

After the last convolutional layer the network has four fully connected layers. (lines 257-261)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (line
248-256). I used a 50% dropout rate because that seemed to give a good training
vs validation error.

The model was trained and validated on different data sets to ensure that the model was not overfitting. I used data from 15 different sets. Some sets were complete laps around the track, and some sets were many short recorded sessions of swerving to correct errant driving.

The model was tested by running it through the simulator and praying that the vehicle could stay on the track. 
Watching the simulator run was like watching a bad horror movie: "DON'T GO IN THERE!" I wanted to shout, "CAN'T YOU SEE YOU'RE DRIVING OFF THE ROAD INTO THE WATER!!!"

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving as close to the center of the road as possible. I used a combination of center lane driving in both directions around the track, recovering from the left and right sides of the road, and many entry paths into the 2 sharp curves just after the bridge.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Since the course suggested the NVIDIA model I read their paper and implemented the model. I thought (correctly) that this would be a safe bet.

My initial NVIDIA model had no dropout layers. Their paper didn't mention any,
so I started with none. But my first runs showed a small training error but a
much larger validation error, implying overfitting, so I added dropout layers
after each of the convolutional layers. I ran a few different dropout rate
experiments and found that 50% gave a good training vs validation error rate.

I used 80% of the data for training and 20% for validation.

Once the training error was close to the validation error, I trained the model on a few clean runs around the track.
The car drove reasonably well with these results, but it drove off the track around the sharper curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. See video.mp4.

See below for further discussion of data sets creation.

#### 2. Final Model Architecture

The final model architecture is the NVIDIA architecture with dropout. The keras `model.summary()` command gives a good visual-ish description of the model:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 70, 318, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 157, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 33, 157, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636       dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 15, 77, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 6, 37, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       dropout_3[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 4, 35, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       dropout_4[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 2, 33, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4224)          0           dropout_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           422500      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
____________________________________________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

I started with 4 clean runs around the track the "right" way, counter-clockwise, and then 1 run the opposite way, clockwise.

I tried to stay in the center of the lane for these runs.

![alt text][image2]

This resulted in reasonable autonomous driving for relatively straight road sections, but the car quickly went off the road when the road curved.

Knowing that the car would need to be able to recover if it got near the road edges, I recorded many small sections of driving where I started on the left and right edge of the road and swerved into the center of the lane.  I focused on the areas where the car had a harder time staying on the road, especially the two curves after the bridge. My hope was this would train the car to steer towards the center of the lane and successfully around corners when it got too close to the edge, and I was right.

These images show a recovery from the right edge back to the center:

![alt text][image3]

![alt text][image4]

![alt text][image5]

The course suggested augmenting the data by flipping the images horizonatlly and using the negative angle from the original image, which made sense to me: these extra images train the network for left and right turns, regardless of what the actual track looks like.

I only used the center camera from the simulator data. I could have added the left and right camera images, but I found it wasn't necessary.

After the collection process, I had 15,347 data points. After horizontal flipping I had 30,694 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used five epochs. I started with five because that's what the course suggested. I experimented with more and fewer epochs but it didn't seem to improve automomous driving. The actual error values didn't seem to correlate with how well the automomous driving did around the track, so it's possible that fewer or more epochs would have been better if I had experimented with it more.
