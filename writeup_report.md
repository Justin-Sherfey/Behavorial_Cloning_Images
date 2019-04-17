# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the files labeled below:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing my results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use a model that consists of a convolution neural network with both 3x3 and 5x5 filter sizes. The model has RELU layers to make it nonlinear and the data is normalized in the model using Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

I use different data augmentation techniques like flipping images horizontally and using left and right images to help generalize the model and prevent overfitting. I also trained and validated it on different sets of data, then I ran it through the simulator to make sure that the car could stay on the road consistently around the loops. 

#### 3. Model parameter tuning

I used used an Adam optimizer for the model, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used myself driving around different parts of the track for training data. I also put in some veering back to the center and other self fixing behaviors to improve the training data. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I found the Nvidia model to be a very good model to follow. https://devblogs.nvidia.com/deep-learning-self-driving-cars/
The lessons recommended this architecture and I found it to be very successful. 

To test the model, I split my steering angle data and image into a validation and training set. The mean squared error was low on both training and validation steps because I was using data augmentation techniques. 

Getting the data augmentation to work well was a bit tricky. One problem I was faced with was applying the data augmentation techniques correctly, because of small errors I was not able to train my model that well and it was not able to drive around the track very well. After fixing this I also created more training data of self fixing behavior so that the model had better data to learn from. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

I modeled my final model architecture after Nvidias, below is a visualization. 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
![Net](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving using the three different angles:

![alt text](https://github.com/Justin-Sherfey/Behavorial_Cloning_Images/blob/master/center_2019_04_14_20_15_06_368.jpg "center")
![alt_text](https://github.com/Justin-Sherfey/Behavorial_Cloning_Images/blob/master/left_2019_04_14_19_24_47_066.jpg "left")
![alt_text](https://github.com/Justin-Sherfey/Behavorial_Cloning_Images/blob/master/right_2019_04_14_19_24_29_439.jpg "right")

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to self fix itself and drive in the center of the road if it ever found itself veering off to the side. These images show what a recovery looks like starting from the right side of the road :

![alt text](https://github.com/Justin-Sherfey/Behavorial_Cloning_Images/blob/master/recovery1.jpg "1")
![alt text](https://github.com/Justin-Sherfey/Behavorial_Cloning_Images/blob/master/recovery2.jpg "2")
![alt text](https://github.com/Justin-Sherfey/Behavorial_Cloning_Images/blob/master/recovery3.jpg "3")

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help and generalize the training data. For example, here are some images that have been converted to rgb, cropped,  and resized:

converted to rgb
![alt text](https://github.com/Justin-Sherfey/Behavorial_Cloning_Images/blob/master/resized_rgb.png "rgb")
cropped
![alt text](https://github.com/Justin-Sherfey/Behavorial_Cloning_Images/blob/master/resized2_cropped.png "cropped")
resized
![alt_text](https://github.com/Justin-Sherfey/Behavorial_Cloning_Images/blob/master/resized.png "resized")

Just this data was enough for me to get the car to drive around the first track proficiently. 

After the collection process, I had 908 number of data points. I then preprocessed this data by cropping, flipping, and resizing the images as to emphasize the important elements of the image. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I could have used much more data to make better training data and thus made my car drive better. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the cars ability. I used an adam optimizer so that manually training the learning rate wasn't necessary.
