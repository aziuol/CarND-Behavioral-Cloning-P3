#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Dense, GlobalAveragePooling2D, ELU, Convolution2D, MaxPooling2D, Activation
from keras.layers import Lambda, Conv2D, BatchNormalization, Dropout, Flatten
from keras.layers.advanced_activations import ELU
from keras.layers.core import Lambda
from keras.models import Model, Sequential, model_from_json, load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import np_utils
from PIL import Image
from scipy import misc
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm
import argparse
import csv
import cv2
import glob
import json
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import tensorflow as tf


# In[2]:


print(os.getcwd())
#recording_path = "C:/Users/louiz/OneDrive/Bureau/term1-simulator-windows/recordings/1_2xlap/"
recording_path = "C:/Users/louiz/OneDrive/Bureau/term1-simulator-windows/recordings/udacity/"
driving_log_path = recording_path + "driving_log.csv"
print(driving_log_path)

df = pd.read_csv(driving_log_path, index_col = False)
df.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']


# In[3]:


df["steer"].mean()
#df["steer"].plot.box(figsize=(6, 6))
df["steer"].describe()

# steering +25' to -25; display precision 25.00.  log precisison to 0.000001
precision =  0.000001

unique_steering_angles = np.unique(df["steer"])
print(f' Unique steering angles" {unique_steering_angles}')
print(f' Number of Unique steering angles" {len(unique_steering_angles)}')

fig = plt.figure(figsize = (15,8))
fig.suptitle("Steering Angle Bins (precision 0.000001)")
hist = df["steer"].hist(bins=len(unique_steering_angles))


# In[4]:


# 0.9500002 is the max.
precision =  0.000001
# list of bins for steering angles measured in training mode.

# list of bins for steering angles measured in training mode.
df["steer"].median()
print(f'mean: {df["steer"].mean()}')
#df["steer"].value_counts()

#print(df["steer"].value_counts().max)


# In[5]:


# Get bin probabilties
bins = np.unique(df["steer"])

target_count = df["steer"].value_counts().mean() 
keep_probs = []
for no, label in enumerate(bins):
    count = df["steer"].value_counts()[label]
    if count < target_count:
        keep_probs.append(1.0)
    else:
        keep_probs.append(1.0/(count/target_count))

#rint(len(keep_probs))
#rint(keep_probs)
#print(len(bins))
#print(f'bins {bins} {len(bins)}')


# In[6]:


df_pruned = df

print(f'Original size: {df.steer.size}')

for i in range(len(bins)):
    indexNames = df[df['steer'] == bins[i]].index
    prune_list = []
    for j in indexNames:
        if np.random.rand() > keep_probs[i]:
            prune_list.append(j)

    df_pruned = df_pruned.drop(prune_list)
    
print(f'Pruned size: {df_pruned.steer.size}')
#print(f'Pruned center_images size: {len(center_images)}')
# steering +25' to -25; display precision 25.00.  log precisison to 0.000001
precision =  0.000001

angles = np.unique(df_pruned["steer"])
print(f'bins {angles}')
print(f'number of bins {len(angles)}')

fig = plt.figure(figsize = (15,8))
fig.suptitle("Steering Angle Bins (precision 0.000001)")
hist = df_pruned["steer"].hist(bins=len(angles))


# In[7]:


center = df_pruned.center.tolist()
center_recover = df_pruned.center.tolist() 
left = df_pruned.left.tolist()
right = df_pruned.right.tolist()
steering = df_pruned.steer.tolist()
steering_recover = df_pruned.steer.tolist()


# In[8]:

#  training data will be centre.  Use a 90/10 split for train/validation testing.
center, steering = shuffle(center, steering)
center, X_valid, steering, y_valid = train_test_split(center, steering, test_size = 0.10, random_state = 100) 


#  Allocate images to appropriate list for right, left and straight steering.  
steer_straight, steer_left, steer_right = [], [], []
angle_straight, angle_left, angle_right = [], [], []
for i in steering:
    # Positive steering angle is right steer, negative angle is left steer
    index = steering.index(i)
    if i > 0.15:
        steer_right.append(center[index])
        angle_right.append(i)
    if i < -0.15:
        steer_left.append(center[index])
        angle_left.append(i)
    else:
        steer_straight.append(center[index])
        angle_straight.append(i)


#  Find the number of samples from straight - left driving to straight - right driving.  
ds_size, dl_size, dr_size = len(steer_straight), len(steer_left), len(steer_right)
main_size = math.ceil(len(center_recover))
l_xtra = ds_size - dl_size
r_xtra = ds_size - dr_size
# Generate random list of indices for left and right - these images classified as recovery images.
indice_L = random.sample(range(main_size), l_xtra)
indice_R = random.sample(range(main_size), r_xtra)

# Adjsutmejnt for steering and re-classification here to steer-right driving
for i in indice_L:
    if steering_recover[i] < -0.15:
        steer_left.append(right[i])
        angle_left.append(steering_recover[i] - 0.27)

# Adjsutmejnt for steering and re-classification here to steer-left driving
for i in indice_R:
    if steering_recover[i] > 0.15:
        steer_right.append(left[i])
        angle_right.append(steering_recover[i] + 0.27)

# combine all steering images for training set with labels. 
X_train = steer_straight + steer_left + steer_right
y_train = np.float32(angle_straight + angle_left + angle_right)

# Augmentation: Generate random brightness function, simulation of night/day driving variations.
def random_brightness(image):
    #Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #Generate new random brightness for night/day condition variances
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img 

# Augmentation: Flip image around vertical axis to create symmetry for right steering and left steering training and balancing the training/validation sets.
def flip(image, angle):
    new_image = cv2.flip(image,1)
    new_angle = angle*(-1)
    return new_image, new_angle

def random_shadow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bright_factor = 0.3
    
    x = random.randint(0, hsv.shape[1])
    y = random.randint(0, hsv.shape[0])

    width = random.randint(int(hsv.shape[1]/2),hsv.shape[1])
    if(x+ width > hsv.shape[1]):
        x = hsv.shape[1] - x
    height = random.randint(int(hsv.shape[0]/2),hsv.shape[0])
    if(y + height > hsv.shape[0]):
        y = hsv.shape[0] - y
    
    #Assuming HSV image
    hsv[y:y+height,x:x+width,2] = hsv[y:y+height,x:x+width,2]*bright_factor

    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img

# Crop image to only include the view of the road and elimate some of the bonnet, resize to 64x64 dimension
def crop_resize(image):
    cropped = cv2.resize(image[60:140,:], (64,64))
    return cropped


# Training generator: shuffles the training data, and randomly selects the data for the batches.
# the generator resizes all data appropriately with a random brightness transform, and applies the flip transform at random.
# Note that teh random_shadow augmentation is not applied due to problems with the model.

def generator_data(batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
        data, angle = shuffle(X_train, y_train)
        for i in range(batch_size):
            choice = int(np.random.choice(len(data),1))
            batch_train[i] = crop_resize(random_brightness(mpimg.imread(data[choice].strip())))
            batch_angle[i] = angle[choice]*(1+ np.random.uniform(-0.10,0.10))
            #Flip random images#
            flip_coin = random.randint(0,1)
            if flip_coin == 1:
                batch_train[i], batch_angle[i] = flip(batch_train[i], batch_angle[i])
            flip_coin = random.randint(0,1)
            # if flip_coin == 1:
            #     batch_train[i] = random_shadow(batch_train[i])

        yield batch_train, batch_angle

# Validation generator: pick random samples. Apply resizing and cropping on chosen samples        
def generator_valid(data, angle, batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
        data, angle = shuffle(data,angle)
        for i in range(batch_size):
            rand = int(np.random.choice(len(data),1))
            batch_train[i] = crop_resize(mpimg.imread(data[rand].strip()))
            batch_angle[i] = angle[rand]

        yield batch_train, batch_angle


# In[13]:


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def train_model(args):
    data_generator = generator_data(128)
    valid_generator = generator_valid(X_valid, y_valid, 128)

    # Training Architecture: inspired by NVIDIA architecture #
    input_shape = (64,64,3)
    #input_shape = (160, 320, 3)
    
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding='same',  kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(80, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(40, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(Dense(1, kernel_regularizer=l2(0.001)))
    adam = Adam(lr = 0.0001)
    
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    if args.m_pretrained != None:
        print(f'using pretrained model')
        model = load_model(args.m_pretrained)
    else:
        #model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
        model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])

    model.summary()
    model.fit_generator(data_generator, 
                        samples_per_epoch = math.ceil(len(X_train)), 
                        nb_epoch=args.nb_epoch, 
                        validation_data = valid_generator, 
                        nb_val_samples = len(X_valid), 
                        callbacks=[checkpoint],
                        verbose=1)

    print('Training Completed')


def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    #parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    #parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    #parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=25)
    #parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    #parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=bool,   default='true')
    #parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    parser.add_argument("-m", help="path to trained model", dest='m_pretrained',      type=str,   default=None)
    args = parser.parse_args()
        
    train_model(args)

if __name__ == '__main__':
    main()