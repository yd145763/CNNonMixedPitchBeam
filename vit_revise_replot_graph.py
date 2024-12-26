# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:44:38 2024

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:33:52 2024

@author: limyu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast

from matplotlib.ticker import StrMethodFormatter
df_vit = pd.read_csv("https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/df_results_consistency.csv", index_col = 0)
df_various = pd.read_csv("https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/df_results_various_models.csv", index_col = 0)

df_cnn = pd.read_csv("https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/df_cnn_results_consistency.csv", index_col = 0)
df_cnn_various = pd.read_csv("https://raw.githubusercontent.com/yd145763/CNNonMixedPitchBeam/refs/heads/main/df_cnn_results_various_models.csv", index_col = 0)

def find_TV_difference(df):
    difference_list = []
    epoch_list = []
    for i in range(len(df)):
        validation_accuracy_list = df['validation_accuracy_list']
        validation_accuracy = validation_accuracy_list[i]
        # Convert the string to a list
        validation_accuracy = ast.literal_eval(validation_accuracy)
        epoch_list.append(len(validation_accuracy))
        validation_accuracy = np.array(validation_accuracy)
        max_index = validation_accuracy.argmax()
        
        training_loss_list = df['training_loss_list']
        training_loss = training_loss_list[i]
        # Convert the string to a list
        training_loss = ast.literal_eval(training_loss)
        training_loss_best = training_loss[max_index]
        
        validation_loss_list = df['validation_loss_list']
        validation_loss = validation_loss_list[i]
        # Convert the string to a list
        validation_loss = ast.literal_eval(validation_loss)
        validation_loss_best = validation_loss[max_index]
        difference = training_loss_best - validation_loss_best 
        
        difference_list.append(difference)
    
    return difference_list, epoch_list

def find_accuracy_consistency(df):

    training_accuracy_best_list = []
    validation_accuracy_best_list = []
    for i in range(len(df)):
        validation_accuracy_list = df['validation_accuracy_list']
        validation_accuracy = validation_accuracy_list[i]
        # Convert the string to a list
        validation_accuracy = ast.literal_eval(validation_accuracy)

        validation_accuracy = np.array(validation_accuracy)
        max_index = validation_accuracy.argmax()
        
        training_accuracy_list = df['training_accuracy_list']
        training_accuracy = training_accuracy_list[i]
        # Convert the string to a list
        training_accuracy = ast.literal_eval(training_accuracy)
        training_accuracy_best = training_accuracy[max_index]
        training_accuracy_best_list.append(training_accuracy_best)
        
        validation_accuracy_list = df['validation_accuracy_list']
        validation_accuracy = validation_accuracy_list[i]
        # Convert the string to a list
        validation_accuracy = ast.literal_eval(validation_accuracy)
        validation_accuracy_best = validation_accuracy[max_index]
        validation_accuracy_best_list.append(validation_accuracy_best)
        
    accuracy = df['accuracy_list']
    accuracy_list = [i for i in accuracy]

    
    return training_accuracy_best_list, validation_accuracy_best_list, accuracy_list

training_accuracy_consistency, validation_accuracy_consistency, testing_accuracy_consistency = find_accuracy_consistency(df_vit)

runs = np.arange(1, len(training_accuracy_consistency)+1, 1)
fig = plt.figure(figsize=(15, 15))
ax = plt.axes()
ax.plot(runs, testing_accuracy_consistency, color = "blue", linewidth = 5)
ax.plot(runs, training_accuracy_consistency, color = "red", linewidth = 5)
ax.plot(runs, validation_accuracy_consistency, color = "green", linewidth = 5)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(60)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(60)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=60)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("Runs")
plt.ylabel("Accuracies")
plt.legend(["Testing Accuracy", "Training Accuracy", "Validation Loss"], prop={'weight': 'bold','size': 50}, loc = "best")
plt.show()
plt.close()



difference_vit, epoch_list_vit = find_TV_difference(df_vit)
difference_cnn, epoch_list_cnn = find_TV_difference(df_cnn)


fig = plt.figure(figsize=(15, 15))
ax = plt.axes()
ax.plot(runs, difference_vit, color = "blue", linewidth = 5)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(60)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(60)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=50)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("Runs")
plt.ylabel("Training - Validation")
plt.show()
plt.close()

# Combine data into a list
data =[difference_vit, difference_cnn]

fig = plt.figure(figsize=(4, 5))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['ViT', 'CNN']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("(Training - Validation)")

# Show the plot
plt.show()

# Extracting minimum and maximum points from the box plot
for i, whisker in enumerate(box['whiskers']):
    if i % 2 == 0:  # even index, represents minimum point
        min_val = whisker.get_ydata()[1]
        print(f'Minimum point for box {i//2 + 1}: {min_val}')
    else:  # odd index, represents maximum point
        max_val = whisker.get_ydata()[1]
        print(f'Maximum point for box {i//2 + 1}: {max_val}')

# Combine data into a list
data =[epoch_list_vit, epoch_list_cnn]

fig = plt.figure(figsize=(4, 5))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['ViT', 'CNN']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Epoch\n(Before Stopping)")

# Show the plot
plt.show()

time_list_vit = df_vit['duration_list']
time_list_cnn = df_cnn['duration_list']
# Combine data into a list
data =[time_list_vit, time_list_cnn]

fig = plt.figure(figsize=(4, 5))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['ViT', 'CNN']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Training duration (s)")

# Show the plot
plt.show()


accuracy_list_vit = df_vit['accuracy_list']
accuracy_list_cnn = df_cnn['accuracy_list']
# Combine data into a list
data =[accuracy_list_vit, accuracy_list_cnn]

fig = plt.figure(figsize=(4, 5))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['ViT', 'CNN']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Accuracy")

# Show the plot
plt.show()


conf = df_vit[['confusion_matrix_list', 'accuracy_list']]
conf_sorted = conf.sort_values(by='accuracy_list', ascending=True)
#conf = conf_sorted['confusion_matrix_list']
conf_sorted = conf_sorted.reset_index(drop = True)
conf = conf_sorted['confusion_matrix_list']
x = conf[199]
print(x)

x = x.replace('[', '').replace(']', '')

conf_matrix = np.array([list(map(int, row.split())) for row in x.strip().split('\n')])

plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="turbo", 
            xticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparse)', 'Region B (sparse)', 'Region C (sparse)'], 
            yticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparse)', 'Region B (sparse)', 'Region C (sparse)'])


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



conf = df_cnn[['confusion_matrix_list', 'accuracy_list']]
conf_sorted = conf.sort_values(by='accuracy_list', ascending=True)
#conf = conf_sorted['confusion_matrix_list']
conf_sorted = conf_sorted.reset_index(drop = True)
conf = conf_sorted['confusion_matrix_list']
x = conf[199]
print(x)

x = x.replace('[', '').replace(']', '')

conf_matrix = np.array([list(map(int, row.split())) for row in x.strip().split('\n')])

plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="turbo", 
            xticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparse)', 'Region B (sparse)', 'Region C (sparse)'], 
            yticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparse)', 'Region B (sparse)', 'Region C (sparse)'])


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


accuracies = df_vit[['training_loss_list', 'validation_loss_list', 'training_accuracy_list','validation_accuracy_list','accuracy_list']]
accuracies_sorted = accuracies.sort_values(by='accuracy_list', ascending=True)
accuracies_sorted = accuracies_sorted.reset_index(drop = True)
accuracies_losses_data = accuracies_sorted.iloc[-1, :]
training_loss = accuracies_losses_data['training_loss_list']
training_loss = ast.literal_eval(training_loss)
validation_loss = accuracies_losses_data['validation_loss_list']
validation_loss = ast.literal_eval(validation_loss)

training_accuracy = accuracies_losses_data['training_accuracy_list']
training_accuracy = ast.literal_eval(training_accuracy)
validation_accuracy = accuracies_losses_data['validation_accuracy_list']
validation_accuracy = ast.literal_eval(validation_accuracy)
epoch = np.arange(1, len(training_loss)+1,1)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.plot(epoch, training_loss, color = "blue", linewidth = 5)
ax.plot(epoch, validation_loss, color = "red", linewidth = 5)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(50)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(50)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=50)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Training loss", "Validation Loss"], prop={'weight': 'bold','size': 40}, loc = "best")
plt.show()
plt.close()

fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.plot(epoch, training_accuracy, color = "blue", linewidth = 5)
ax.plot(epoch, validation_accuracy, color = "red", linewidth = 5)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(50)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(50)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=50)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"], prop={'weight': 'bold','size': 40}, loc = "best")
plt.show()
plt.close()

df_various['percentage'] = [i*100 for i in df_various['accuracy_list']]
mat = df_various.pivot('N_DENSE_UNITS_list', 'N_LAYERS_list', 'percentage')
mat_list = mat.values.tolist()

plt.figure(figsize=(6, 4))
ax = sns.heatmap(mat_list, annot=True, cmap='turbo', fmt=".2f")

mat.columns
ax.set_xticklabels(mat.columns, fontweight="bold", fontsize = 16)
ax.set_yticklabels(mat.index, fontweight="bold", fontsize = 16)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Accuracy (%)", fontweight="bold", fontsize = 16)
font = {'color': 'black', 'weight': 'bold', 'size': 16}
ax.set_ylabel("MLP Head Size", fontdict=font)
ax.set_xlabel("Transformer Encorders", fontdict=font)
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(16)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
plt.show()
plt.close()

df_cnn_various['percentage'] = [i*100 for i in df_cnn_various['accuracy_list']]
mat = df_cnn_various.pivot('layer_size_list', 'num_conv_list', 'percentage')
mat_list = mat.values.tolist()

plt.figure(figsize=(6, 4))
ax = sns.heatmap(mat_list, annot=True, cmap='turbo', fmt=".2f")

mat.columns
ax.set_xticklabels(mat.columns, fontweight="bold", fontsize = 16)
ax.set_yticklabels(mat.index, fontweight="bold", fontsize = 16)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Accuracy (%)", fontweight="bold", fontsize = 16)
font = {'color': 'black', 'weight': 'bold', 'size': 16}
ax.set_ylabel("Dense Layers Size", fontdict=font)
ax.set_xlabel("CNN Layers", fontdict=font)
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(16)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
plt.show()
plt.close()


df_various_filtered = df_various[df_various['N_LAYERS_list'] ==1]
df_various_filtered = df_various_filtered[df_various_filtered['N_DENSE_UNITS_list'] ==128]
x = df_various_filtered['confusion_matrix_list']
x = x[0]

x = x.replace('[', '').replace(']', '')

conf_matrix = np.array([list(map(int, row.split())) for row in x.strip().split('\n')])

plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="turbo", 
            xticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparse)', 'Region B (sparse)', 'Region C (sparse)'], 
            yticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparse)', 'Region B (sparse)', 'Region C (sparse)'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 20}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Classification", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", fontsize = 13)
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", fontsize = 13)
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(20)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')

plt.show()
plt.close()

df_cnn_various.columns
df_cnn_various_filtered = df_cnn_various[df_cnn_various['num_conv_list'] ==1]
df_cnn_various_filtered = df_cnn_various_filtered[df_cnn_various_filtered['layer_size_list'] ==128]
x = df_cnn_various_filtered['confusion_matrix_list']
x = x[0]

x = x.replace('[', '').replace(']', '')

conf_matrix = np.array([list(map(int, row.split())) for row in x.strip().split('\n')])

plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="turbo", 
            xticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparse)', 'Region B (sparse)', 'Region C (sparse)'], 
            yticklabels=['Region A (focused)', 'Region B (focused)', 'Region C (focused)', 'Region A (sparse)', 'Region B (sparse)', 'Region C (sparse)'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 20}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Classification", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", fontsize = 13)
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", fontsize = 13)
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(20)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')

plt.show()
plt.close()