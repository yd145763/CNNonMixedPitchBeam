
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time  # For timing operations
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.metrics import confusion_matrix  # For evaluating classification performance
import seaborn as sns  # For visualizing the confusion matrix
from keras.models import Sequential  # For building the CNN model
from keras.layers import (Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout)  # Model layers
from keras.utils import to_categorical  # For one-hot encoding labels
from tensorflow.keras.callbacks import EarlyStopping  # For stopping the model early during training



dataset = []
label = []

url_0 = "https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/reactive_focused/"
paths_0 = pd.read_csv(url_0+'file_list.csv')
paths_0 = [i for i in paths_0.columns]
for file in paths_0:
    print(url_0+file)
    df = pd.read_csv(url_0+file, index_col = 0)
    df = np.array(df)
    image = np.reshape(df, (64, 64, 1))
    dataset.append(image)
    label.append(0)

url_1 = "https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/nearfield_focused/"
paths_1 = pd.read_csv(url_1+'file_list.csv')
paths_1 = [i for i in paths_1.columns]
for file in paths_1:
    print(url_1+file)
    df = pd.read_csv(url_1+file, index_col = 0)
    df = np.array(df)
    image = np.reshape(df, (64, 64, 1))
    dataset.append(image)
    label.append(1)

url_2 = "https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/farfield_focused/"
paths_2 = pd.read_csv(url_2+'file_list.csv')
paths_2 = [i for i in paths_2.columns]
for file in paths_2:
    print(url_2+file)
    df = pd.read_csv(url_2+file, index_col = 0)
    df = np.array(df)
    image = np.reshape(df, (64, 64, 1))
    dataset.append(image)
    label.append(2)

url_3 = "https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/reactive_sparse/"
paths_3 = pd.read_csv(url_3+'file_list.csv')
paths_3 = [i for i in paths_3.columns]
for file in paths_3:
    print(url_3+file)
    df = pd.read_csv(url_3+file, index_col = 0)
    df = np.array(df)
    image = np.reshape(df, (64, 64, 1))
    dataset.append(image)
    label.append(3)

url_4 = "https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/nearfield_sparse/"
paths_4 = pd.read_csv(url_4+'file_list.csv')
paths_4 = [i for i in paths_4.columns]
for file in paths_4:
    print(url_4+file)
    df = pd.read_csv(url_4+file, index_col = 0)
    df = np.array(df)
    image = np.reshape(df, (64, 64, 1))
    dataset.append(image)
    label.append(4)

url_5 = "https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/farfield_sparse/"
paths_5 = pd.read_csv(url_5+'file_list.csv')
paths_5 = [i for i in paths_5.columns]
for file in paths_5:
    print(url_5+file)
    df = pd.read_csv(url_5+file, index_col = 0)
    df = np.array(df)
    image = np.reshape(df, (64, 64, 1))
    dataset.append(image)
    label.append(5)




training_loss_list = []
training_accuracy_list = []
validation_loss_list = []
validation_accuracy_list = []
confusion_matrix_list = []
accuracy_list = []
duration_list = []
prediction_time_list = []


for count in range(2):

    num_conv = 1
    layer_size = 128

    # Define the EarlyStopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',  # Monitor training loss
        min_delta=0,  # Minimum change to qualify as an improvement
        patience=30,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # Verbosity mode
        mode='max',  # Maximize the monitored quantity
        restore_best_weights=True  # Whether to restore model weights to the best observed during training
    )
    drop = 0.2
    model = None
    model = Sequential()
    model.add(Convolution2D(int(image.shape[0]/2), (3, 3), input_shape = (image.shape[0], image.shape[1], image.shape[2]), activation = 'relu', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(drop))
    for _ in range(num_conv-1):
        model.add(Convolution2D(int(image.shape[0]/2), (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
        model.add(BatchNormalization(axis = -1))
        model.add(Dropout(drop))
    model.add(Flatten())
    layer_size_v = layer_size
    model.add(Dense(activation = 'relu', units=layer_size_v))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(drop))
    model.add(Dense(activation = 'relu', units=layer_size_v))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(drop))

    model.add(Dense(activation = 'sigmoid', units=6))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])





    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.40, random_state = 0)



    # Define the EarlyStopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',  # Monitor training loss
        min_delta=0,  # Minimum change to qualify as an improvement
        patience=30,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # Verbosity mode
        mode='max',  # Maximize the monitored quantity
        restore_best_weights=True  # Whether to restore model weights to the best observed during training
    )

    start_time = time.time()
    history = model.fit(np.array(X_train), 
                            y_train, 
                            batch_size = 5, 
                            verbose = 1, 
                            epochs = 100,      #Changed to 3 from 50 for testing purposes.
                            validation_split = 0.5,
                            shuffle = True,
                            callbacks=[early_stopping_callback]
                        )
    end_time = time.time()

    duration = end_time - start_time
    duration_list.append(duration)


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

    training_accuracy = history.history['accuracy']
    training_accuracy_list.append(training_accuracy)
    validation_accuracy = history.history['val_accuracy']
    validation_accuracy_list.append(validation_accuracy)




    pred_start = time.time()
    # Predict the test set results
    y_pred = model.predict(np.array(X_test))
    pred_end = time.time()
    prediction_time = pred_end - pred_start
    prediction_time_list.append(prediction_time)
    
    # Convert predictions to binary classes
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Convert true labels to binary classes
    y_true = np.argmax(np.array(y_test), axis=1)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    confusion_matrix_list.append(conf_matrix)

    total_sum = np.sum(conf_matrix)
    correct_answer = conf_matrix[0,0] + conf_matrix[1,1] +conf_matrix[2,2]+conf_matrix[3,3]+conf_matrix[4,4]+conf_matrix[5,5]
    accuracy = correct_answer/total_sum
    accuracy_list.append(accuracy)

    print(count)
    print("Accuracy: {:.2f}%".format(accuracy*100))
    

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="turbo", 
                xticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparsed)', 'Region B (sparsed)', 'Region C (sparsed)'], 
                yticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparsed)', 'Region B (sparsed)', 'Region C (sparsed)'])


    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")

    font = {'color': 'black', 'weight': 'bold', 'size': 12}
    ax.set_ylabel("Actual", fontdict=font)
    ax.set_xlabel("Classification", fontdict=font)

    # Setting tick labels bold
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
    #ax.tick_params(axis='both', labelsize=12, weight='bold')
    for i, text in enumerate(ax.texts):
        text.set_fontsize(12)
    for i, text in enumerate(ax.texts):
        text.set_fontweight('bold')

    plt.show()
    plt.close()

df_results = pd.DataFrame()
df_results['training_loss_list'] = training_loss_list
df_results['validation_loss_list'] = validation_loss_list
df_results['training_accuracy_list'] = training_accuracy_list
df_results['validation_accuracy_list'] = validation_accuracy_list
df_results['duration_list'] = duration_list
df_results['accuracy_list'] = accuracy_list
df_results['confusion_matrix_list'] = confusion_matrix_list
df_results['prediction_time_list'] = prediction_time_list

df_results.to_csv('/home/grouptan/Documents/yudian/ViT_revised/df_cnn_results_consistency.csv')
