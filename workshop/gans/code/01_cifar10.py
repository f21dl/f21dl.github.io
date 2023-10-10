import tensorflow as tf # pip install tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np 


# using Keras to load dataset 
(X_train, y_train),(X_test, y_test) = keras.datasets.cifar10.load_data()
print("X_train shape=", X_train.shape," X_test shape =", X_test.shape)

fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    

# scale the pixel intensities from 0 to 255 to [0,1] range
X_train = X_train.astype("float32")/255.0

# create create a dataset to iterate through images 
batch_size=128
dataset=tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset=dataset.batch(batch_size, drop_remainder=True).prefetch(1)

plt.show()


