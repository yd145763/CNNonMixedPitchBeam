# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:09:47 2024

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:10:00 2024

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
num_head_list = []
num_trans_list = []
mlp_size_list = []
accuracy_list = []
training_loss_list = []
validation_loss_list = []
time_list = []
confusion_matrix_list = []


from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential



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

CONFIGURATION = {
    "BATCH_SIZE": 16,
    "IM_SIZE": 64,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 5,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 256,
    "NUM_CLASSES": 3,
    "PATCH_SIZE": 8,
    "PROJ_DIM": 192,
    "CLASS_NAMES": ["reactive", "nearfield", "farfield"],
}

import tensorflow as tf### models
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime
import pathlib
import io
import os
import time
import random
from PIL import Image
import tensorflow_datasets as tfds
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.regularizers  import L2, L1
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature

class PatchEncoder(Layer):
  def __init__(self, N_PATCHES, HIDDEN_SIZE):
    super(PatchEncoder, self).__init__(name = 'patch_encoder')

    self.linear_projection = Dense(HIDDEN_SIZE)
    self.positional_embedding = Embedding(N_PATCHES, HIDDEN_SIZE )
    self.N_PATCHES = N_PATCHES

  def call(self, x):
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        strides=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        rates=[1, 1, 1, 1],
        padding='VALID')

    patches = tf.reshape(patches, (tf.shape(patches)[0], CONFIGURATION["PATCH_SIZE"]**2, patches.shape[-1]))

    embedding_input = tf.range(start = 0, limit = self.N_PATCHES, delta = 1 )
    output = self.linear_projection(patches) + self.positional_embedding(embedding_input)

    return output

class TransformerEncoder(Layer):
  def __init__(self, N_HEADS, HIDDEN_SIZE):
    super(TransformerEncoder, self).__init__(name = 'transformer_encoder')

    self.layer_norm_1 = LayerNormalization()
    self.layer_norm_2 = LayerNormalization()

    self.multi_head_att = MultiHeadAttention(N_HEADS, HIDDEN_SIZE )

    self.dense_1 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)
    self.dense_2 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)

  def call(self, input):
    x_1 = self.layer_norm_1(input)
    x_1 = self.multi_head_att(x_1, x_1)

    x_1 = Add()([x_1, input])

    x_2 = self.layer_norm_2(x_1)
    x_2 = self.dense_1(x_2)
    output = self.dense_2(x_2)
    output = Add()([output, x_1])

    return output

class ViT(Model):
  def __init__(self, N_HEADS, HIDDEN_SIZE, N_PATCHES, N_LAYERS, N_DENSE_UNITS):
    super(ViT, self).__init__(name = 'vision_transformer')
    self.N_LAYERS = N_LAYERS
    self.patch_encoder = PatchEncoder(N_PATCHES, HIDDEN_SIZE)
    self.trans_encoders = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
    self.dense_1 = Dense(N_DENSE_UNITS, tf.nn.gelu)
    self.dense_2 = Dense(N_DENSE_UNITS, tf.nn.gelu)
    self.dense_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation = 'softmax')
  def call(self, input, training = True):

    x = self.patch_encoder(input)

    for i in range(self.N_LAYERS):
      x = self.trans_encoders[i](x)
    x = Flatten()(x)
    x = self.dense_1(x)
    x = self.dense_2(x)

    return self.dense_3(x)

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping

# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='loss',  # Monitor training loss
    min_delta=0,  # Minimum change to qualify as an improvement
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,  # Verbosity mode
    mode='min',  # Minimize the monitored quantity
    restore_best_weights=True  # Whether to restore model weights to the best observed during training
)


            
N_HEADS = 2
N_LAYERS = 1
N_DENSE_UNITS = 256

X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.40, random_state = 0)

num_head_list.append(N_HEADS)
num_trans_list.append(N_LAYERS)
mlp_size_list.append(N_DENSE_UNITS)

vit = ViT(N_HEADS = N_HEADS, HIDDEN_SIZE = (CONFIGURATION["PATCH_SIZE"]**2)*3, N_PATCHES = CONFIGURATION["PATCH_SIZE"]**2,
    N_LAYERS =N_LAYERS, N_DENSE_UNITS = N_DENSE_UNITS)
vit(tf.zeros([2,CONFIGURATION["PATCH_SIZE"]**2,CONFIGURATION["PATCH_SIZE"]**2,3]))

vit.summary()
vit.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#loss_function = CategoricalCrossentropy()
#metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k=2, name = "top_k_accuracy")]
#vit.compile(optimizer = Adam(learning_rate = CONFIGURATION["LEARNING_RATE"]), 
#            loss = loss_function,
#            metrics = metrics,)

start_time = time.time()
history = vit.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 5, 
                         verbose = 1, 
                         epochs = 100,      #Changed to 3 from 50 for testing purposes.
                         validation_split = 0.2,
                         shuffle = True,
                         callbacks=[early_stopping_callback]
                     )
end_time = time.time()
time_spent = end_time - start_time
time_list.append(time_spent)
print("Test_Accuracy: {:.2f}%".format(vit.evaluate(np.array(X_test), np.array(y_test))[1]*100))
accuracy = vit.evaluate(np.array(X_test), np.array(y_test))[1]*100



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

training_loss = history.history['loss']
training_loss_list.append(training_loss)
validation_loss = history.history['val_loss']
validation_loss_list.append(validation_loss)


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict the test set results
y_pred = vit.predict(np.array(X_test))
# Convert predictions to binary classes
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert true labels to binary classes
y_true = np.argmax(np.array(y_test), axis=1)
  
# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

confusion_matrix_list.append(conf_matrix)
total_sum = np.sum(conf_matrix)
correct_answer = conf_matrix[0,0] + conf_matrix[1,1] +conf_matrix[2,2]
accuracy = correct_answer/total_sum
accuracy_list.append(accuracy)

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
plt.title("Confusion Matrix"+"_"+"HEADS"+str(N_HEADS)+"_"+"LAYERS"+str(N_LAYERS)+"_"+"MLP"+str(N_DENSE_UNITS)+"_"+"ImageSize"+str(SIZE)
          +"\n"+"Accuracy"+str(accuracy))
plt.show()
plt.close()
            
            
            
df_results = pd.DataFrame()
df_results['num_head_list'] =num_head_list
df_results['num_trans_list'] =num_trans_list
df_results['mlp_size_list'] =mlp_size_list
df_results['accuracy_list'] =accuracy_list
df_results['training_loss_list'] =training_loss_list
df_results['validation_loss_list'] =validation_loss_list
df_results['time_list'] = time_list
df_results['confusion_matrix_list'] = confusion_matrix_list
df_results.to_csv(image_directory+'trans_df_results.csv')

sns.pairplot(df_results)
plt.show()

df_results.columns
correlations = df_results[['num_conv_list', 'num_dense_list', 'layer_size_list', 'drop_list', 'accuracy_list']].corr()
correlations.to_csv(image_directory+'trans_correlations.csv')


correlation_with_accuracy = correlations['accuracy_list'][['num_conv_list', 'num_dense_list', 'layer_size_list', 'drop_list']]
correlation_with_accuracy.to_csv(image_directory+'trans_correlation_with_accuracy.csv')
print(correlation_with_accuracy)

# Plotting the correlation heatmap
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()