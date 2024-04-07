# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:13:52 2024

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 07:28:36 2024

@author: limyu
"""


import numpy as np


import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras
import pandas as pd 
import statistics

os.environ['KERAS_BACKEND'] = 'tensorflow' 
num_conv_list = []
num_dense_list = []
layer_size_list = []
accuracy_list = []
SIZE_list = []
mean_difference_list = []
last_difference_list = []
drop_list = []

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential

num_conv = 2
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


#Apply CNN
# ### Build the model

#############################################################
###2 conv and pool layers. with some normalization and drops in between.



model = None
model = Sequential()
model.add(Convolution2D(SIZE/2, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(drop))
for _ in range(num_conv-1):
    model.add(Convolution2D(SIZE/2, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(drop))
model.add(Flatten())
layer_size_v = layer_size
for _ in range(num_dense):
    model.add(Dense(activation = 'relu', units=layer_size_v))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(drop))
    layer_size_v = int(round(layer_size_v/2, 0))
    if layer_size_v < 2:
        break
model.add(Dense(activation = 'sigmoid', units=3))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

    
###############################################################    
    
 ### Split the dataset
# 
# I split the dataset into training and testing dataset.
# 1. Training data: 80%
# 2. Testing data: 20%
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.40, random_state = 0)


# When training with Keras's Model.fit(), adding the tf.keras.callback.TensorBoard callback 
# ensures that logs are created and stored. Additionally, enable histogram computation 
#every epoch with histogram_freq=1 (this is off by default)
#Place the logs in a timestamped subdirectory to allow easy selection of different training runs.

#import datetime

#log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# ### Training the model
# As the training data is now ready, I will use it to train the model.   

#Fit the model
history = model.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 5, 
                         verbose = 1, 
                         epochs = 5,      #Changed to 3 from 50 for testing purposes.
                         validation_split = 0.2,
                         shuffle = True
                      #   callbacks=callbacks
                     )

# ## Accuracy calculation
# 
# I'll now calculate the accuracy on the test data.

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))
accuracy = model.evaluate(np.array(X_test), np.array(y_test))[1]*100
accuracy_list.append(accuracy)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict the test set results
y_pred = model.predict(np.array(X_test))
# Convert predictions to binary classes
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert true labels to binary classes
y_true = np.argmax(np.array(y_test), axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Reactive', 'Nearfield', 'Farfield'], 
            yticklabels=['Reactive', 'Nearfield', 'Farfield'])
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Region A', 'Region B', 'Region C'], 
            yticklabels=['Region A', 'Region B', 'Region C'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Predicted", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
plt.title("Confusion Matrix"+"_"+"Conv"+str(num_conv)+"_"+"Dense"+str(num_dense)+"_"+"LayerSize"+str(layer_size)+"_"+"ImageSize"+str(SIZE)
          +"\n"+"Accuracy"+str(accuracy))
plt.show()
plt.close()
