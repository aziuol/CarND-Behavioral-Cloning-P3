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

[NVIDIA_Model]: ./examples/NVIDIA_Model.png "NVIDIA CNN architecture"
[image2]: ./examples/placeholder.png "Grayscaling"
[steer_right1]: ./examples/steer_right1.png "Steer Right example1- autonomous mode"
[steer_right2]: ./examples/steer_right2.png "Steer Right example 2- autonomous mode"
[steer_left1]: ./examples/steer_left1.png "Steer Left example 1- autonomous mode"
[steer_left2]: ./examples/steer_left2.png "Steer Left example 2- autonomous mode"
[steer_straight1]: ./examples/steer_straight1.png "Steer straight example 1- autonomous mode"
[steer_straight2]: ./examples/steer_straight2.png "Steer straight example 2- autonomous mode"
[train_straight]: ./examples/train_straight.png "Steer straight example- training mode"
[run1.mp4]: ./examples/run1.mp4 "Lakeside driving - autonomous mode"
[Image_Balancing]: ./examples/balancing_training_data.jpg "Pre and post balancing of training dataset"

[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[End to End Learning for Self-Driving Cars]: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `model.py`                   | Implementation of the nvidea model architecture and executes training of the model |
| `model.h5`                   | The model  weights of the CNN 														|
| `Pre-processing_model.ipynb` | Implementation of the preprocessing steps used for data exploration / visualisations of the data sets |
| `drive.py`                   | This implements the neccessary communication callbacks for hooking the model predictions to the simuluator to drive the car in autonomous mode.  |


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

- Balancing of the dataset.  
- Data Pre-processing 


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My convolutional neural network (CNN) model is based on NVIDIA's [CNN architecture detailed in their paper here][End to End Learning for Self-Driving Cars] and implemented using pythons Keras API framework.

Just like the NVIDIA model, the network is a regression model (not a classification model) and we train the weights to minimise the mean squared error of the steering output.

The network consists of a total of 9 layers.  The first layer is an imbuilt lamda normalisation layer included a pre-processing step of the data).

Following the lamda layer is 5 convolutional layers with Kernal size of 5x5 for the first 3 layers, and 3x3 for the last 2 layers.  For the first 3 layers, the stride is 2x2, and for the last 2 the depth remains at 64 (non-strided layers).
The filter depths for the CNN layers are 24, 36, 48, 64, 64 respectively.
Between each layer, a RELU activation is also implemented to avoid overfitting and introduce non-linearity.

After the 5th convolutional layer, the data is flattened and there are 4 fully connected layers with the kernal regularizer that drives outlier weights close to 0 (such that these features are not actually eliminated) and also a drop out of 50% betweem each connected layer, again to reduce overfitting.

The below table is a summary of the CNN implementation.

Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 64, 64, 3)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 24)        1824
_________________________________________________________________
activation_1 (Activation)    (None, 30, 30, 24)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 13, 36)        21636
_________________________________________________________________
activation_2 (Activation)    (None, 13, 13, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 5, 48)          43248
_________________________________________________________________
activation_3 (Activation)    (None, 5, 5, 48)          0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 3, 64)          27712
_________________________________________________________________
activation_4 (Activation)    (None, 3, 3, 64)          0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 1, 64)          36928
_________________________________________________________________
activation_5 (Activation)    (None, 1, 1, 64)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 64)                0
_________________________________________________________________
dense_1 (Dense)              (None, 80)                5200
_________________________________________________________________
dropout_1 (Dropout)          (None, 80)                0
_________________________________________________________________
dense_2 (Dense)              (None, 40)                3240
_________________________________________________________________
dropout_2 (Dropout)          (None, 40)                0
_________________________________________________________________
dense_3 (Dense)              (None, 16)                656
_________________________________________________________________
dropout_3 (Dropout)          (None, 16)                0
_________________________________________________________________
dense_4 (Dense)              (None, 10)                170
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 140,625
Trainable params: 140,625
Non-trainable params: 0

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was also trained on a minimal data set of 542 centre images of a balanced dataset.  The original 2x lap recording created 3856 images (inclusive of left and right) but in order to prevent overfitting of "straight steering" control output these images were removed so that the CNN learnt how to steer towards the centre of the road.

The model was validated on different data (the data was split with a 90:10 trainig/validation ratio) and by running the simulator it was confirmed that there was no overfitting occuring as the vehicle steered when required and remained on the track.  

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.


#### 4. Appropriate training data

Initially 5 laps were recorded oand this was used as training/validation data for the model.  The outcome was not good, the car didn't know how to steer appropriately across the first corner.  After many attempts I  passed the first corner, but then the next hurdle was the bridge... it was pretty clear at this point that the network had not trained correctly to steer when required.  I then trained 10 laps of data, but of course the result was the same and I began investigating the training data.

By visualising the steering angles (45 bins in total), it was clear that the training data was unbalanced with the straight steering (0) bin beeing many magnitudes higher that would confirm an a bias in the network.  With such a strong bias in the input training data, the network will not train correctly despite how good the model architecture is. 

I resolved this issue by "pruning" the input data, implementing a weighted ratio for any bins that required pruning. 
`for no, label in enumerate(bins):
    count = df["steer"].value_counts()[label]
    if count < target_count:
        keep_probs.append(1.0)
    else:
        keep_probs.append(1.0/(count/target_count))`

Where `count` is the frequency count for a steering bin, and `target_count` is the mean frequency count across the bins.  


Augmentation was also performed on the training data.  The benefits are sumarised in the below table:such as random brightness (night/day driving), flip (left and right steering), random shadows (robustness in ignoring the shadow features) where also included to add variance and robustness to the training features.   Importantly, the input data was cropped to only include the road data (removing horizon and sky data) so that the road curvature would be learned in the feature map as a parameter for the steering.   Perhaps a more agressive cropping strategy could be used to focus the learning to "immediate" steering as stated by other students that the curvature near the horizon is future steering information.

| File                          |Description                                                                                                   |
| ------------------------------|--------------------------------------------------------------------------------------------------------------|
| Image croping				    | Important in successful training to ensure model learns the ROI of the road for active immediate steering.   | 
|								| The road curvature in horizon is for future steering and not for immediate safe active steering.             |
|								| The CNN steering output is not a time series vector, each image is randomly shuffled etc our architecture has|
|                               | no memory. A much more advanced architecture LSTM will need to be looked at that has not been covered in the |
|                               | in the course yet.                                                                                           |
| Data Augmentation - flip      | Used balance left and right steering for unbias learning									                   |
| Data Augmentation - brightness| Random brighness to generalise training data for sunny/dark  condition and robustness for steering features  | 																											
| Data Augmentation	- shadows   | Random augmentation to darken image segments simulating shadows for robustness in steering featrure          |
| Using Multiple Cameras        | Not implemented.  Not neccessary for successful training.  Used Kera's batch generator instead!     		   |	

One of the most important aspects of the implementation was to implement a batch generator as recommended in the course to constantly feed image batches for training and validation.  This was very important as it allowed an optimisation in the training pipeline as a mini-batch generation is handled by the multi-core CPU and then passed to the GPU.  This is an effective way of ensuring the GPU(s) hardware is performing at maximum capacity, reducing time spent on generating the model. 

Another import aspect in the implementation was to include model checkpoints.  This was valuable as I could save  multiple  models that met the model metric accuracy.

I was able to experiment with different datasets such as "wobble driving" 2x full laps of recovery driving, and also testing the model in new simulator environment (e.g jungle track with no training).  I found that with 2x full laps of wobble driving, there was model was bias had learnt too wobble steer!  As I had my checkpoints, I did not have to start from scratch.

The model is successful in steering robustly in the lakeside track, a minimum of 1 epochs is required for correct steering and successful driving.

![Image Balancing is shown here, prior to balancing of the dataset a large portion of the dataset was straight steering][balancing_training_data.jpg]
As discussed earlier, the straight steering sample set was pruned to balance the data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement the an already proven CNN model architecture for the self driving car. 

There were other architectures such as the Comma AI model, the original VGG model could also be used.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

As stated earlier, to combat overfitting the model included drop-out layers between each fully connected layer and RELU.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, such as sthe first sharp turn and also the bridge.  To improve this behaviour the balancing of the dataset was very useful as was the data augmentation.  I still have issues with the shadows -> although I have not trained with the jungle track, the vehicle control clearly cannot handle shadow features in the map.

At the end of the process, the vehicle is able to drive autonomously around the lakeside track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`model.py` `lines 18-24`) consisted of a convolution neural network with the following layers and layer sizes discussed above.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![An example of the recording for straight steering that was used for training][[train_straight].

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer to the center.
I did this using a "wobble" driving method, it was not effective as I needed to remove the steering drift so to speak.  In the end, I did not need recovery data because my 2x laps contained recovery in there or close calls and I had pruned and balanced my dataset.

I did not repeat the process on track 2, as I wanted to test he model performance in a brand new environment that it had not been trained on.  It actually performed OK considering this, note that the speed had to be limited to 9 mph.  It did not perform well with dark shadows -> there was too much variation in the dataset and test set for it to perform and it would fail on a specific shadowy corner.    The next step would be to visualise the internal layers of teh CNN to understand the feature map parameters and if the model can be generalised well with a small  input data set.
If not I will then add training with track 2.

Here are some examples of driving behaviour in autonomous mode.  
![straight steering][steer_straight1.png]  Note that this is a screenshot taken while running the simulation.
![Another example of straigh steering performed by the model][steer_straight2.png]
![Left steering] [steer_left1.png]
![Left steering] [steer_left2.png]
![Right steering] [steer_right1.png]
![Right steering] [steer_left2.png]

![Video of the test simulation is here in the lakeside track.] [https://youtu.be/B2P188iXLto]

#### 4. Acknowledgements and closing remarks
The next challenge will be to prepare a model capable of driving autonomously on the jungle track. 
I would also try to implement some autonomous driving for brake and accelerator control, for this I will need to train on different road surfaces.

I would like to thank fellow Udacity students, who provided valuable insights on some of the problems posed in this project.

#### 5. References

[1]  Jeremy-Shannon, CarND-Behavioural-Cloning-Project,  https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project, 2017
[2]  David A. Ventimiglia, CarND-Behavioral-Cloning, https://github.com/dventimi/CarND-Behavioral-Cloning, 2017
[3]  An Nguyen, SDC-P3 BehavouralCloning, https://github.com/ancabilloni/SDC-P3-BehavioralCloning, 2017

