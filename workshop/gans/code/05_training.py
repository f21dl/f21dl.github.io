
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
batchSize=128
dataset=tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset=dataset.batch(batchSize, drop_remainder=True).prefetch(1)

# plt.show()




# coding size - the dimension of the input vector for the generator
codingSize = 100

def buildGenerator( codingSize=100 ):
    generator = tf.keras.Sequential()
    
    # latent variable as input 
    generator.add(keras.layers.Dense(1024, activation="relu", input_shape=(codingSize,) ) )
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.Dense(1024, activation="relu") )
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.Dense(128*8*8, activation="relu") )
    generator.add(keras.layers.Reshape((8,8,128) ) )
    assert generator.output_shape == (None, 8, 8, 128) # None is the batch size

    generator.add(keras.layers.Conv2DTranspose(filters=128,kernel_size=2, strides=2, activation="relu", padding="same") )
    assert generator.output_shape == (None,16,16,128)
    generator.add(keras.layers.BatchNormalization() )
    
    generator.add(keras.layers.Conv2DTranspose(filters=3, kernel_size=2, strides=2, activation="tanh", padding="same") )
    assert generator.output_shape == (None, 32, 32, 3)
    generator.add(keras.layers.BatchNormalization() )
    
    return generator 


generator = buildGenerator()
nbrImgs = 3

def plotGeneratedImages( nbrImgs, titleadd="" ):
    noise = tf.random.normal( [nbrImgs, 100] )
    imgs = generator.predict(noise)
    
    fig = plt.figure(figsize=(40,10))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1,nbrImgs,i+1)
        ax.imshow( (img*255).astype(np.uint8) )
    fig.suptitle( "Gen images"+titleadd, fontsize=25 )
    plt.show()


# plotGeneratedImages( nbrImgs )

# the discriminator
def buildDiscriminator():
    discriminator = tf.keras.Sequential()
    
    discriminator.add(keras.layers.Conv2D(filters=64,kernel_size=3,strides=2,activation=keras.layers.LeakyReLU(0.2), padding="same", 
                                                                                                                     input_shape=(32,32,3) ) )
    discriminator.add(keras.layers.Conv2D(filters=128,kernel_size=3, strides=2, activation=keras.layers.LeakyReLU(0.2),padding="same") )
    discriminator.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation=keras.layers.LeakyReLU(0.2), padding="same") )
    discriminator.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, activation=keras.layers.LeakyReLU(0.2), padding="same" ) )
    
    # classifier
    discriminator.add(keras.layers.Flatten() )
    discriminator.add(keras.layers.Dropout(0.4) )
    discriminator.add(keras.layers.Dense(1,activation="sigmoid") )
    return discriminator

discriminator = buildDiscriminator()
# compile our model 
opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
discriminator.trainable = False 


gan = keras.models.Sequential( [ generator, discriminator ] )
# compile the gan
opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
gan.compile( loss="binary_crossentropy", optimizer=opt )

# Comine images into 'gif' (store animations)
from PIL import Image
import cv2 # pip install opencv-python
images = []

def animatedGif():
    noise_1 = tf.random.normal(shape=[4,codingSize] )
    imgs = generator.predict(noise_1)
    img0 = (imgs[0]*255).astype(np.uint8)
    img1 = (imgs[1]*255).astype(np.uint8)
    img2 = (imgs[2]*255).astype(np.uint8)
    img3 = (imgs[3]*255).astype(np.uint8)
    
    img = cv2.hconcat([img0, img1, img2, img3])
    img = Image.fromarray(np.uint8(img)).convert("RGB")
    return img 


print('----')
def trainGAN(gan, dataset, batchSize, codingsSize, nEpochs):
    generator, discriminator = gan.layers
    for epoch in range( nEpochs ):
        for X_batch in dataset:
            # phase 1 - training discriminator
            noise = tf.random.normal(shape=[batchSize, codingsSize] )
            generatedImages = generator.predict(noise)
            X_fake_and_real = tf.concat( [ generatedImages, X_batch], axis=0 )
            y1 = tf.constant( [[0.0]]*batchSize + [[1.0]]*batchSize )
            discriminator.trainable=True 
            d_loss_accuracy = discriminator.train_on_batch( X_fake_and_real, y1 )
            
            # phase 2 - training the generator 
            noise = tf.random.normal( shape=[batchSize, codingsSize] )
            y2 = tf.constant( [[1.0]] * batchSize )
            discriminator.trainable = False 
            g_loss = gan.train_on_batch( noise, y2 )
            
        print("epoch:",epoch," d_loss_accuracy:",d_loss_accuracy, "g_loss:", g_loss )
        plotGeneratedImages( 3, titleadd=":Epoch{}".format(epoch) )
        # create animated gif 
        img = animatedGif()
        images.append(img)
        print("---")


nEpochs = 100
trainGAN( gan, dataset, batchSize, codingSize, nEpochs )

# create gif - images at each epoch 
images[0].save('./genImages.gif', save_all=True, append_Images=images[1:], optimize=False, duration=500, loop=0)





    




