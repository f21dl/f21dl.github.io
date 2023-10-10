
import tensorflow as tf # pip install tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np 


# coding size - the dimension of the input vector for the generator
codingSize = 100

def buildGenerator( codingSize=100 ):
    generator = tf.keras.Sequential()
    
    # latent variable as input 
    generator.add(keras.layer.Dens(1024, activation="relu", input_shape=(codingSize,) ) )
    generator.add(keras.layer.BatchNormalization())
    generator.add(keras.layers.Dense(1024, activation="relu") )
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.Dense(128*8*8, ativation="relu") )
    generator.add(keras.layers.Reshape((8,8,128) ) )
    assert generator.output_shape == (None, 8, 8, 128) # None is the batch size

    generator.add(keras.layers.Conv2DTranspose(filters=128,kernel_size=2, strides=2, activation="relu", padding="same") )
    assert generator.output_shape == (None,16,16,128)
    generator.add(keras.layers.BatchNormalization() )
    
    generator.add(keras.layers.Conv2DTranspose(filters=3, kernel_size=2, strides=2, activation="tanh", padding="same") )
    assert generator.output_shape == (None, 32, 32, 3)
    generator.add(keras.layers.BatchNormalization() )
    
    return generator 


