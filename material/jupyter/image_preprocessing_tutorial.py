
"""
image_preprocessing_tutorial.py

This tutorial is intended to give a basis introduction on how to preprocess 
image data so that they are in standard pandas-dataframe format.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
from sklearn.model_selection import train_test_split

# Directory contains jpg-images
image_dir = 'D:/Universitaet/F21DL_21_22/ExampleImages'
# New shape for images:
new_size = (256, 256)

images, images_bw = [], []
label_strs = []
for filename in os.listdir(image_dir):
    print(f'\n-> {filename}')
    
    # Read in jpg file as array:
    image_wc = cv2.imread(f'{image_dir}/{filename}', cv2.IMREAD_COLOR)
    # shape = (rows, columns, color channels)
    print(f'image_wc: {image_wc.shape}, {type(image_wc)}')
    
    # cv2 reads images as BGR and not RGB:
    image_tc = cv2.cvtColor(src=image_wc, code=cv2.COLOR_BGR2RGB)
    # Convert images to grey scale:
    image_tc_bw = cv2.cvtColor(src=image_wc, code=cv2.COLOR_BGR2GRAY)
    print(f'\nimage_tc_bw: {image_tc_bw.shape}, {type(image_tc_bw)}')
    
    # Resize images to the new shape:
    image_tc_rs = cv2.resize(src=image_tc, dsize=new_size)
    image_tc_bw_rs = cv2.resize(src=image_tc_bw, dsize=new_size)
    
    # Store images and labels:
    images.append(image_tc_rs)
    images_bw.append(image_tc_bw_rs)
    label_strs.append(filename.split('_')[0])
    
fig, axis = plt.subplots(nrows=1, ncols=3)
axis[0].imshow(image_wc)
axis[0].set_title('BGR: Wrong color')
axis[1].imshow(image_tc)
axis[1].set_title('RGB: True color')
axis[2].imshow(image_tc_bw, cmap='Greys')
axis[2].set_title('Grey-scale')
plt.show()

# Stack images to a single array:
images_arr = np.stack(images)
print(f'\nimages_arr: {images_arr.shape}')
images_bw_arr = np.stack(images_bw)
print(f'images_bw_arr: {images_bw_arr.shape}')

fig, axis = plt.subplots(nrows=1, ncols=3)
axis[0].imshow(images_bw_arr[0], cmap='Greys')
axis[1].imshow(images_bw_arr[1], cmap='Greys')
axis[2].imshow(images_bw_arr[2], cmap='Greys')
plt.show()

# Label strings to unique integers:
print(f'\nlabel_strs: {label_strs}')
label_ints = pd.Categorical(label_strs).codes
print(f'label_ints: {label_ints}')
labels_arr = np.array(label_ints)
print(f'labels_arr: {labels_arr}, {labels_arr.shape}, {type(labels_arr)}')

# Putting it all together:
# Flattening array along columns; 
# 256*256 = 65536 / with RGB 256*256*3 = 196608
features = images_bw_arr.reshape((len(labels_arr), -1))
print(f'\nfeatures: {features.shape}')
labels = labels_arr.reshape((len(labels_arr), -1))
print(f'labels: {labels.shape}')

df = pd.DataFrame(
    data=features, 
    columns=['pixels_'+str(i+1) for i in range(features.shape[1])])
print(f'\ndf: {df.shape}')
print(df.head())
# Append labels-column:
df['labels'] = labels
print(f'df: {df.shape}')

# Train test split of the data:
# 1) If you work with two numpy arrays:
x_train, x_test, y_train, y_test = train_test_split(features, labels)

# 1) If you work with a pandas dataframe:
# IMPORTANT: If features and labels are stored in the same dataframe,
# you need to remove those when passing the features, just as done
# below with the slicing df.iloc[:,0:-1], otherwise you have 'label leackage'
# https://en.wikipedia.org/wiki/Leakage_(machine_learning)
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,0:-1], df['labels'])

