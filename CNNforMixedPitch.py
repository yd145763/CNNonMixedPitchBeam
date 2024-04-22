# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 07:28:36 2024

@author: limyu
"""


import numpy as np

#Set the `numpy` pseudo-random generator at a fixed value
#This helps with repeatable results everytime you run the code. 
np.random.seed(1000)

import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras
import pandas as pd 
import statistics
import time

os.environ['KERAS_BACKEND'] = 'tensorflow' # Added to set the backend as Tensorflow
#We can also set it to Theano if we want. 
num_conv_list = []
num_dense_list = []
layer_size_list = []
accuracy_list = []
training_loss_list = []
validation_loss_list = []
time_list = []
confusion_matrix_list = []


from tensorflow.keras.callbacks import EarlyStopping

# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='loss',  # Monitor training loss
    min_delta=0,  # Minimum change to qualify as an improvement
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,  # Verbosity mode
    mode='min',  # Minimize the monitored quantity
    restore_best_weights=False  # Whether to restore model weights to the best observed during training
)

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential

num_convS = 1,2,3,4
num_denseS = 1,2,3,4
layer_sizeS =  128,256,512,1024


SIZE = 64
for num_conv in num_convS:
    for num_dense in num_denseS:
        for layer_size in layer_sizeS:


            num_conv_list.append(num_conv)
            num_dense_list.append(num_dense)
            layer_size_list.append(layer_size)


            drop = 0.2


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
            model.add(Convolution2D(int(SIZE/2), (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu', data_format='channels_last'))
            model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
            model.add(BatchNormalization(axis = -1))
            model.add(Dropout(drop))
            for _ in range(num_conv-1):
                model.add(Convolution2D(int(SIZE/2), (3, 3), activation = 'relu'))
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
            start_time = time.time()
            #Fit the model
            history = model.fit(np.array(X_train), 
                                     y_train, 
                                     batch_size = 5, 
                                     verbose = 1, 
                                     epochs = 100,      #Changed to 3 from 50 for testing purposes.
                                     validation_split = 0.2,
                                     shuffle = True,
                                     callbacks=[early_stopping_callback]
                                  #   callbacks=callbacks
                                 )
            end_time = time.time()
            time_spent = end_time - start_time
            time_list.append(time_spent)
            # ## Accuracy calculation
            # 
            # I'll now calculate the accuracy on the test data.
            
            print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))

            
            
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
            
            train_loss = history.history['loss']

            
            val_loss = history.history['val_loss']
            training_loss_list.append(train_loss)
            validation_loss_list.append(val_loss)
            
            
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
            plt.title("Confusion Matrix"+"_"+"Conv"+str(num_conv)+"_"+"Dense"+str(num_dense)+"_"+"LayerSize"+str(layer_size)+"_"+"ImageSize"+str(SIZE)
                      +"\n"+"Accuracy"+str(accuracy))
            plt.show()
            plt.close()


df_results = pd.DataFrame()
df_results['num_conv_list'] = num_conv_list 
df_results['num_dense_list'] = num_dense_list
df_results['layer_size_list'] = layer_size_list
df_results['accuracy_list'] = accuracy_list
df_results['training_loss_list'] = training_loss_list
df_results['validation_loss_list'] = validation_loss_list
df_results['time_list'] = time_list
df_results['confusion_matrix_list'] = confusion_matrix_list

df_results.to_csv(image_directory+'df_results.csv')

import seaborn as sns 

sns.pairplot(df_results)
plt.show()

df_results.columns
correlations = df_results[['num_conv_list', 'num_dense_list', 'layer_size_list', 'drop_list', 'accuracy_list']].corr()
correlations.to_csv(image_directory+'correlations.csv')


correlation_with_accuracy = correlations['accuracy_list'][['num_conv_list', 'num_dense_list', 'layer_size_list', 'drop_list']]
correlation_with_accuracy.to_csv(image_directory+'correlation_with_accuracy.csv')
print(correlation_with_accuracy)

# Plotting the correlation heatmap
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
################################################
### ANOTHER WAY TO DEFINE THE NETWORK using Sequential model
#Sequential 
#You can create a Sequential model by passing a list of layer instances to the constructor:
"""
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential

model = None
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(activation = 'relu', units=512))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Dense(activation = 'relu', units=256))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Dense(activation = 'sigmoid', units=2))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

INPUT_SHAPE = (SIZE, SIZE, 3)   #change to (SIZE, SIZE, 3)
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
drop1 = keras.layers.Dropout(rate=0.2)(norm1)
conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis = -1)(pool2)
drop2 = keras.layers.Dropout(rate=0.2)(norm2)

flat = keras.layers.Flatten()(drop2)  #Flatten the matrix to get it ready for dense.

hidden1 = keras.layers.Dense(512, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1)
drop3 = keras.layers.Dropout(rate=0.2)(norm3)
hidden2 = keras.layers.Dense(256, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis = -1)(hidden2)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)

out = keras.layers.Dense(2, activation='sigmoid')(drop4)   #units=1 gives error

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
                loss='categorical_crossentropy',   #Check between binary_crossentropy and categorical_crossentropy
                metrics=['accuracy'])
print(model.summary())

"""

