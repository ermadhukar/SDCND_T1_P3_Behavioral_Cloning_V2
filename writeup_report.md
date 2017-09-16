

#**Behavioral Cloning Writeup** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report





## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results



####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```




####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Input data is being normalised and cropped with Keras.
NVIDIA model has been refered for this model. It has of 5 convolution layers and 4 fully-connected layers.
First 3 convolution layers has 5x5 filters with stride value of 2. Other two layers has 3x3 filters with regular stride value of 1. To introduce non-linearity, convolution layers are activated by RELU function.
First 3 dense layers are activated with RELU functions and last output layer is activated by a linear function.


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I used the first track for data collection in low resolution and fast mode.

For details about how I created the training data, see the next section. 



###Model Architecture and Training Strategy

####1. Solution Design Approach

Initially results of simple convolutional neural network model were not good. Even after initial balancing mean-square errors were accumulating and leading to undesired steering of the vehicle. Augmenting the balanced dataset make the model able to visualize better angles of the road, which helped in recovering from error accumulation.Since this model had problem with data complexity, nvidia model has been refered. To prevent overfitting dropouts had been added after the convolution layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track initially or stopped in between. To improve the driving behavior in these cases, I changed the training data with bigger sample and trained model with different parameters.

At the end of the process, the vehicle is able to drive autonomously around the track. It is running fine most part of the track without much difficulty. It is also able to steer well around corners and take corrective action to bring car back to center. Though there is some issue in driving the vehicle in exact center throughout the path. I believe with better training data, it can be overcomed.

####2. Final Model Architecture

The final model architecture...

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to               

==================================================

cropping2d_2 (Cropping2D)        (None, 80, 320, 3)    0           cropping2d_input_2[0][0]         
____________________________________________________________________________________________________
lambda_2 (Lambda)                (None, 80, 320, 3)    0           cropping2d_2[0][0]               
____________________________________________________________________________________________________
c1 (Convolution2D)               (None, 38, 158, 24)   1824        lambda_2[0][0]                   
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 38, 158, 24)   0           c1[0][0]                         
____________________________________________________________________________________________________
c2 (Convolution2D)               (None, 17, 77, 36)    21636       dropout_6[0][0]                  
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 17, 77, 36)    0           c2[0][0]                         
____________________________________________________________________________________________________
c3 (Convolution2D)               (None, 7, 37, 48)     43248       dropout_7[0][0]                  
____________________________________________________________________________________________________
dropout_8 (Dropout)              (None, 7, 37, 48)     0           c3[0][0]                         
____________________________________________________________________________________________________
c4 (Convolution2D)               (None, 5, 35, 64)     27712       dropout_8[0][0]                  
____________________________________________________________________________________________________
dropout_9 (Dropout)              (None, 5, 35, 64)     0           c4[0][0]                         
____________________________________________________________________________________________________
c5 (Convolution2D)               (None, 3, 33, 64)     36928       dropout_9[0][0]                  
____________________________________________________________________________________________________
dropout_10 (Dropout)             (None, 3, 33, 64)     0           c5[0][0]                         
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 6336)          0           dropout_10[0][0]                 
____________________________________________________________________________________________________
d1 (Dense)                       (None, 100)           633700      flatten_2[0][0]                  
____________________________________________________________________________________________________
d2 (Dense)                       (None, 50)            5050        d1[0][0]                         
____________________________________________________________________________________________________
d3 (Dense)                       (None, 10)            510         d2[0][0]                         
____________________________________________________________________________________________________
out (Dense)                      (None, 1)             11          d3[0][0]            

===================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
____________________________________________________________________________________________________
Train on 7013 samples, validate on 370 samples
Epoch 1/5
818s - loss: 0.0542 - val_loss: 0.0445
Epoch 2/5
672s - loss: 0.0422 - val_loss: 0.0422
Epoch 3/5
628s - loss: 0.0406 - val_loss: 0.0413
Epoch 4/5
574s - loss: 0.0396 - val_loss: 0.0393
Epoch 5/5
539s - loss: 0.0387 - val_loss: 0.0385



'''

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

To run the simulator smooth, I have opted to run the model in fastest mode with low resolution. Also, since I was training the model on my laptop, I didnt want to overload my cpu with a lot of data. It has resulted in not very good trained model but I got the idea of the process and going forward with some iteration, I will be able to train and run the model perfectly.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its path while driving. These recorded data has been used for training the model.

![Left Cam Image](img/capt_img.png) 

To augment the data set, I also flipped images and angles thinking that this would help the model to steer the vehicle in both clockwise and anti-clockwise direction.

![cent Cam Image](img/flip1.png)
![flip Cam Image](img/flip2.png)

After the collection process, I preprocessed them by cropping and normalizing. I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5, as after that MSE was not improving much. I used an adam optimizer so that manually training the learning rate wasn't necessary.



```python

```


