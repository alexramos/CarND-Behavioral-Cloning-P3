#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/example1.jpg "Example 1"
[image2]: ./examples/example2_left.jpg "Example 2 Left"
[image3]: ./examples/example2_center.jpg "Example 2 Center"
[image4]: ./examples/example2_right.jpg "Example 2 Right"
[image5]: ./examples/example3.jpg "Example 3"
[image6]: ./examples/example3_flipped.jpg "Example 3 Flipped"

## Rubric Points 
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 video file of an autonomous lap around track 1
* video_2.mp4 video file of an autonomous lap around track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is an adoptation of the convolutional neural network employed by NVIDIA to control steering commands ([paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)) (model.py lines 98-112).

It consists of 5 convolutional layers followed by 4 fully-connected dense layers.  For the first three convolution layers, I use a 5x5 filter size with 2x2 strides and filter depths of 24, 36, and 48.  For the last two convolution layers, I use a non-strided 3x3 filter size and depth of 64.  The model includes ELU (Exponential Linear Unit) layers to introduce nonlinearity. 

Prior to the convolutional layers, the data is normalized in the model using a Keras lambda layer and then cropped to removed 70 and 25 pixels from the top and bottom of the image, respectively. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 114-118). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I also tested the model with dropout layers but this resulted in the car running off the track in track 2.

#### 3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually (model.py line 112).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a dataset of center lane driving consisting of three laps driving each on track 1, track 1 reversed, track 2, and track 2 reversed.  I futher supplemented this data with additional driving data on two particular hairpin turns in track 2 that were giving the model difficulty.

Images from all three cameras were used.  To augment the data I also flipped each image horizontally to yield twice as much training data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to iterate over various architecture decisions, data preprocessing steps, and data augmentations until arriving at a model that could successfully complete a lap around track 1 and then track 2.

My first step was to use a convolutional neural network model similar to the NVIDIA architecture described [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).  I thought this model might be appropriate because it has been used successfully to control steering from camera images on actual self-driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set after 5 epochs of training.  This implied that the model was overfitting.  To combat the overfitting, I modified the model so that it terminated training after three epochs.

The next step was to run the simulator to see how well the car was driving around track one. On its first run, the car would hug the side of the road and then eventually drive off the road before the bridge section.  To improve the driving behavior in these cases, I further trained the model with images from the left and right cameras together with a steering bias correction.

In order to get the car to drive successfully around the 2nd track, I had to add additional data from track 1 reversed, track 2, and track 2 reversed.  I also switched to ELU activation layers from the RELU layers I had used in my initial implemention

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 98-112) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 rgb image   						| 
| Lambda 				| normalize image 								|
| Cropping 				| crop image 									|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 78x158x24 	|
| ELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 37x77x36 	|
| ELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 17x37x48 	|
| ELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 15x35x64 	|
| ELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x33x64 	|
| ELU					|												|
| Flatten				|												|
| Fully connected		| input = 27456, output = 100					|
| Fully connected		| input = 100, output = 50 						|
| Fully connected		| input = 50, output = 10 						|
| Fully connected		| input = 10, output = 1 						|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![example1][image1]

In order to get the car to steer back towards the center if it drifts off the center of the road, I added images from the left and right cameras of the car in combination with a steering correction of 0.2 and -0.2 for left and right cameras, respectively.  These images below show example images from left, center, and right cameras:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.  Lastly, I further collected data by driving on each of the tracks in reverse for three laps.

To augment the data sat, I also flipped images horizontally and angles thinking that this would double my traing dataset size.  For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 135,000 data points. I then preprocessed this data by normalizing each image (x / 255.0 - 0.5). I also cropped out the top and bottom portions of each image to prevent the model from learning irrelevant features present in the distance or the bottom pixels showing the car's body.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under-fitting. The ideal number of training epochs was 5 as evident but no further gains in validation loss after this point.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
