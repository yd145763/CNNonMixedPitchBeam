# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:15:47 2024

@author: limyu
"""


# https://youtu.be/ho6JXE3EbZ8
"""
@author: Sreenivas Bhattiprolu

Copying VGG16 architecture and picking the conv layers of interest 
to generate filtered responses. 
"""

import numpy as np


import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras
import pandas as pd 
import statistics
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.models import Model

num_conv = 1
num_dense = 1
layer_size =  128
drop = 0.2
SIZE = 64

image_directory = 'C:\\Users\\limyu\\Google Drive\\CNNBeamProfiles\\'
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

reactive_images = os.listdir(image_directory + 'reactive\\')
for i, image_name in enumerate(reactive_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory + 'reactive\\' + image_name)
        print(image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

nearfield_images = os.listdir(image_directory + 'nearfield\\')
for i, image_name in enumerate(nearfield_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory + 'nearfield\\' + image_name)
        print(image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

farfield_images = os.listdir(image_directory + 'farfield\\')
for i, image_name in enumerate(farfield_images):
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory + 'farfield\\' + image_name)
        print(image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(2)

model = None
model = Sequential()
model.add(Convolution2D(4, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(drop))
for _ in range(num_conv-1):
    model.add(Convolution2D(4, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(drop))
model.add(Flatten())
layer_size_v = layer_size

if layer_size_v <= 1:
    model.add(Dense(activation = 'relu', units=layer_size_v))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(drop))   
if layer_size_v >1:
    for _ in range(num_dense):
        print(layer_size_v)
        model.add(Dense(activation = 'relu', units=layer_size_v))
        model.add(BatchNormalization(axis = -1))
        model.add(Dropout(drop))
        layer_size_v = int(round(layer_size_v/2, 0))
        
        if layer_size_v < 2:
            break
model.add(Dense(activation = 'sigmoid', units=3))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())
    
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.40, random_state = 0)


# ### Training the model
# As the training data is now ready, I will use it to train the model.   
from tensorflow.keras.callbacks import EarlyStopping
# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='loss',  # Monitor training loss
    min_delta=0,  # Minimum change to qualify as an improvement
    patience=30,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,  # Verbosity mode
    mode='min',  # Minimize the monitored quantity
    restore_best_weights=True  # Whether to restore model weights to the best observed during training
)


#Fit the model
history = model.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 5, 
                         verbose = 1, 
                         epochs = 100,      #Changed to 3 from 50 for testing purposes.
                         validation_split = 0.5,
                         shuffle = True,
                         callbacks=[early_stopping_callback]
                     )

 
#Understand the filters in the model 
#Let us pick the first hidden layer as the layer of interest.
layer = model.layers #Conv layers at 1, 3, 6, 8, 11, 13, 15
filters, biases = model.layers[0].get_weights()
print(layer[1].name, filters.shape)

   
# plot filters

fig1=plt.figure(figsize=(6, 6))
columns = 2
rows = 2
n_filters = columns * rows
for i in range(1, n_filters +1):
    f = filters[:, :, :, i-1]
    print(f)
    print("")
    fig1 =plt.subplot(rows, columns, i)
    fig1.set_xticks([])  #Turn off axis
    fig1.set_yticks([])
    plt.imshow(f[:, :, 0], cmap='gray') #Show only the filters from 0th channel (R)
    #ix += 1
plt.show()    

#### Now plot filter outputs    

#Define a new truncated model to only include the conv layers of interest
#conv_layer_index = [1, 3, 6, 8, 11, 13, 15]
conv_layer_index = np.arange(0, num_conv*4+num_dense*3+2, 1) #TO define a shorter model
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = Model(inputs=model.inputs, outputs=outputs)
print(model_short.summary())
print(model.summary())



#Input shape to the model is 224 x 224. SO resize input image to this shape.
from keras.preprocessing.image import load_img, img_to_array

image = cv2.imread("C:\\Users\\limyu\\Google Drive\\CNNBeamProfiles\\reactive\\grating12_11pitch2_8_10.03.png")
image = Image.fromarray(image, 'RGB')
image = image.resize((SIZE, SIZE))

# convert the image to an array
img = img_to_array(image)
# expand dimensions to match the shape of model input
img = np.expand_dims(img, axis=0)

# Generate feature output by predicting on the input image
feature_output = model_short.predict(img)

import tensorflow as tf
image = cv2.imread("C:\\Users\\limyu\\Google Drive\\CNNBeamProfiles\\reactive\\grating12_11pitch2_8_10.03.png")
image = Image.fromarray(image, 'RGB')
image = image.resize((SIZE, SIZE))
img = np.array(image)


# Add batch dimension to the input image
img = tf.expand_dims(img, axis=0)
img = np.array(img)


for i in conv_layer_index:
    print(model.layers[i])

for i in conv_layer_index[0:num_conv*4]:
    print(model.layers[i])
    img = model.layers[i](img)
    img = np.array(img)
    fig=plt.figure(figsize=(6, 6))
    for i in range(1, columns*rows +1):
        fig =plt.subplot(rows, columns, i)
        fig.set_xticks([])  #Turn off axis
        fig.set_yticks([])
        plt.imshow(img[0, :, :, i-1], cmap='rainbow')
        #pos += 1
    plt.show()
    plt.close()
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt

img = model.layers[num_conv*4](img)
img = np.array(img)
img = img.transpose()
n = np.arange(0, len(img), 1)

fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(n, img, s = 10, color = 'red', alpha = 1)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Data")
plt.ylabel("Values")
#plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "best")
plt.show()
plt.close()

img = img.transpose()
    
for i in conv_layer_index[(num_conv*4)-1:conv_layer_index[-1]+1]:
    print(model.layers[i])
    img = model.layers[i](img)
    img = np.array(img)
    img = img.transpose()
    n = np.arange(0, len(img), 1)
    fig = plt.figure(figsize=(7, 4))
    ax = plt.axes()
    ax.scatter(n, img, s = 10, color = 'red', alpha = 1)
    #graph formatting     
    ax.tick_params(which='major', width=2.00)
    ax.tick_params(which='minor', width=2.00)
    ax.xaxis.label.set_fontsize(15)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(15)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    #ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.xlabel("Data")
    plt.ylabel("Values")
    #plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "best")
    plt.show()
    plt.close()
    img = img.transpose()

"""

columns = 4
rows = 8
for ftr in feature_output[0:4]:
    #pos = 1
    fig=plt.figure(figsize=(9, 12))
    for i in range(1, columns*rows +1):
        fig =plt.subplot(rows, columns, i)
        fig.set_xticks([])  #Turn off axis
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i-1], cmap='rainbow')
        #pos += 1
    plt.show()

I = 5,6,7,8
for i in I:
    data = feature_output[i].transpose()
    n = np.arange(0, len(data),1)
    plt.scatter(n, data)
    plt.show()
    plt.close()

#==================================================================

import tensorflow as tf
image = cv2.imread("C:\\Users\\limyu\\Google Drive\\CNNBeamProfiles\\reactive\\grating12_11pitch2_8_10.03.png")
image = Image.fromarray(image, 'RGB')
image = image.resize((SIZE, SIZE))
img = np.array(image)

# Add batch dimension to the input image
img = tf.expand_dims(img, axis=0)
img = np.array(img)

output_after_conv2d = model.layers[0](img)
output_after_conv2d = np.array(output_after_conv2d)

fig=plt.figure(figsize=(6, 12))
for i in range(1, columns*rows +1):
    fig =plt.subplot(rows, columns, i)
    fig.set_xticks([])  #Turn off axis
    fig.set_yticks([])
    plt.imshow(output_after_conv2d[0, :, :, i-1], cmap='rainbow')
    #pos += 1
plt.show()

output_after_maxpool2d = model.layers[1](output_after_conv2d)
output_after_maxpool2d = np.array(output_after_maxpool2d)

fig=plt.figure(figsize=(6, 12))
for i in range(1, columns*rows +1):
    fig =plt.subplot(rows, columns, i)
    fig.set_xticks([])  #Turn off axis
    fig.set_yticks([])
    plt.imshow(output_after_maxpool2d[0, :, :, i-1], cmap='rainbow')
    #pos += 1
plt.show()

output_after_batchnorm = model.layers[2](output_after_maxpool2d)
output_after_batchnorm = np.array(output_after_batchnorm)

fig=plt.figure(figsize=(6, 12))
for i in range(1, columns*rows +1):
    fig =plt.subplot(rows, columns, i)
    fig.set_xticks([])  #Turn off axis
    fig.set_yticks([])
    plt.imshow(output_after_batchnorm[0, :, :, i-1], cmap='rainbow')
    #pos += 1
plt.show()

output_after_drop = model.layers[3](output_after_batchnorm)
output_after_drop = np.array(output_after_drop)

fig=plt.figure(figsize=(6, 12))
for i in range(1, columns*rows +1):
    fig =plt.subplot(rows, columns, i)
    fig.set_xticks([])  #Turn off axis
    fig.set_yticks([])
    plt.imshow(output_after_drop[0, :, :, i-1], cmap='rainbow')
    #pos += 1
plt.show()

output_after_flatten = model.layers[4](output_after_drop)
output_after_flatten = np.array(output_after_flatten)
output_after_flatten = output_after_flatten.transpose()
n = np.arange(0, len(output_after_flatten), 1)
plt.scatter(n,output_after_flatten, s=0.1)
plt.show()

output_after_dense = model.layers[5](output_after_flatten.transpose())
output_after_dense = np.array(output_after_dense)
output_after_dense = output_after_dense.transpose()
n = np.arange(0, len(output_after_dense), 1)
plt.scatter(n,output_after_dense, s=5)
plt.show()

output_after_dense1 = model.layers[6](output_after_dense.transpose())
output_after_dense1 = np.array(output_after_dense1)
output_after_dense1 = output_after_dense1.transpose()
n = np.arange(0, len(output_after_dense1), 1)
plt.scatter(n,output_after_dense1, s=5)
plt.show()

output_after_dropout1 = model.layers[7](output_after_dense1.transpose())
output_after_dropout1 = np.array(output_after_dropout1)
output_after_dropout1 = output_after_dropout1.transpose()
n = np.arange(0, len(output_after_dropout1), 1)
plt.scatter(n,output_after_dropout1, s=5)
plt.show()

prediction = model.layers[8](output_after_dropout1.transpose())
prediction = np.array(prediction)
prediction = prediction.transpose()
n = np.arange(0, len(prediction), 1)
plt.scatter(n,prediction, s=10)
plt.show()

#===================================
"""
