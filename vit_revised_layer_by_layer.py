

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import StrMethodFormatter


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
import tensorflow as tf### models
import numpy as np### math computations




import time


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Dense,Flatten, Add,Embedding,LayerNormalization, MultiHeadAttention)


from tensorflow.keras.callbacks import EarlyStopping





CONFIGURATION = {
    "BATCH_SIZE": 16,
    "IM_SIZE": 64,
    "NUM_CLASSES": 6,
    "PATCH_SIZE": 8,
    "N_PATCHES": 64,
    "HIDDEN_SIZE": 64,  
}

class PatchEncoder(Layer):
  def __init__(self, N_PATCHES, HIDDEN_SIZE):
    super(PatchEncoder, self).__init__(name='patch_encoder')
    self.linear_projection = Dense(HIDDEN_SIZE)
    self.positional_embedding = Embedding(N_PATCHES, HIDDEN_SIZE)
    self.N_PATCHES = N_PATCHES

  def call(self, x):
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        strides=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        rates=[1, 1, 1, 1],
        padding='VALID')
    patches = tf.reshape(patches, (tf.shape(patches)[0], CONFIGURATION["N_PATCHES"], -1))
    embedding_input = tf.range(start=0, limit=self.N_PATCHES, delta=1)
    output = self.linear_projection(patches) + self.positional_embedding(embedding_input)
    return output

class TransformerEncoder(Layer):
  def __init__(self, N_HEADS, HIDDEN_SIZE):
    super(TransformerEncoder, self).__init__(name='transformer_encoder')
    self.layer_norm_1 = LayerNormalization()
    self.layer_norm_2 = LayerNormalization()
    self.multi_head_att = MultiHeadAttention(N_HEADS, HIDDEN_SIZE)
    self.dense_1 = Dense(HIDDEN_SIZE, activation=tf.nn.gelu)
    self.dense_2 = Dense(HIDDEN_SIZE, activation=tf.nn.gelu)

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
    super(ViT, self).__init__(name='vision_transformer')
    self.N_LAYERS = N_LAYERS
    self.patch_encoder = PatchEncoder(N_PATCHES, HIDDEN_SIZE)
    self.trans_encoders = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
    self.dense_1 = Dense(N_DENSE_UNITS, tf.nn.gelu)
    self.dense_2 = Dense(N_DENSE_UNITS, tf.nn.gelu)
    self.dense_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation='softmax')

  def call(self, input, training=True):
    x = self.patch_encoder(input)
    for i in range(self.N_LAYERS):
      x = self.trans_encoders[i](x)
    x = Flatten()(x)
    x = self.dense_1(x)
    x = self.dense_2(x)
    return self.dense_3(x)

N_HEADS = 1
N_LAYERS = 1
N_DENSE_UNITS = 128

vit = ViT(N_HEADS=N_HEADS, HIDDEN_SIZE=CONFIGURATION["HIDDEN_SIZE"], N_PATCHES=CONFIGURATION["N_PATCHES"],
          N_LAYERS=N_LAYERS, N_DENSE_UNITS=N_DENSE_UNITS)

# Initialize the model with the new input shape (40x40x1)
vit(tf.zeros([2, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 1]))

vit.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
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
history = vit.fit(np.array(X_train), 
                        y_train, 
                        batch_size = 5, 
                        verbose = 1, 
                        epochs = 100,      #Changed to 3 from 50 for testing purposes.
                        validation_split = 0.5,
                        shuffle = True,
                        callbacks=[early_stopping_callback]
                    )


df = pd.read_csv("https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/nearfield_sparse/grating012umpitch05dutycycle30um_28.7.csv", index_col = 0)  
df = np.array(df)
image = np.reshape(df, (64, 64, 1))
img_with_batch = tf.expand_dims(image, axis=0)

# Visualize the image after the patch encoder
output_after_patch_encoder = vit.layers[0](img_with_batch) #patch encoder by layer index
output_array = output_after_patch_encoder.numpy()

fig = plt.figure(figsize=(4, 4))
ax = plt.axes()
cp=ax.imshow(output_array[0], cmap = 'turbo')
colorbarmax = output_array[0].max().max()
clb=fig.colorbar(cp, cmap = 'turbo', pad = 0.01)
clb.ax.set_title('Values', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
#plt.imshow(img[0], cmap = 'turbo', norm=LogNorm())  # Assuming batch size is 1
plt.axis('off')
plt.show()
plt.close()


# Visualize the image after each Transformer encoder

output_after_transformer0 = vit.layers[1](output_after_patch_encoder)
output_array = output_after_transformer0.numpy()
fig = plt.figure(figsize=(4, 4))
ax = plt.axes()
cp=ax.imshow(output_array[0], cmap = 'turbo')
colorbarmax = output_array[0].max().max()
clb=fig.colorbar(cp, cmap = 'turbo', pad = 0.01)
clb.ax.set_title('Values', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
#plt.imshow(img[0], cmap = 'turbo', norm=LogNorm())  # Assuming batch size is 1
plt.axis('off')
plt.show()
plt.close()


np_output_after_transformer0 = np.array(output_after_transformer0)
np_output_after_transformer0 = np_output_after_transformer0.reshape(1, 64*64)
values = np_output_after_transformer0[0,:]
n = np.arange(0, len(values), 1)
fig = plt.figure(figsize=(4, 4))
ax = plt.axes()
ax.scatter(n, values, s = 1, color = 'red', alpha = 1)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
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


flattened_output_after_transformer0 = tf.expand_dims(np_output_after_transformer0, axis=0)
output_after_transformer1 = vit.layers[2](flattened_output_after_transformer0)
output_array = output_after_transformer1.numpy()
values = output_array[0,0,:]
n = np.arange(0, len(values), 1)
fig = plt.figure(figsize=(4, 4))
ax = plt.axes()
ax.scatter(n, values, s = 10, color = 'red', alpha = 1)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
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




output_after_transformer2 = vit.layers[3](output_after_transformer1)
output_array = output_after_transformer2.numpy()
values = output_array[0,0,:]
n = np.arange(0, len(values), 1)
fig = plt.figure(figsize=(4, 4))
ax = plt.axes()
ax.scatter(n, values, s = 10, color = 'red', alpha = 1)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
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



output_after_dense0 = vit.layers[4](output_after_transformer2)
output_array = output_after_dense0.numpy()
output_array = output_array.transpose() 
n = np.arange(0, len(output_array), 1)
fig = plt.figure(figsize=(4, 4))
ax = plt.axes()
ax.scatter(n, output_array, s = 20, color = 'red', alpha = 1)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
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
