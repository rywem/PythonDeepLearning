# -*- coding: utf-8 -*-

# Part 1 - Building the Convolutional Neural Network

import time
start = time.time()

# Check gpu
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

'''
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
'''
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


batch = 32

with tf.device('/cpu:0'):
    # Initialize the CNN
    classifier = Sequential()
    # Add convolutional layer
    classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    # Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    ## Add additional Convolutional layer to increase accuracy
    classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Flattening
    classifier.add(Flatten())
    # Full connection
    ## Input Layer
    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    ## Output later
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the CNN to the images
    
    ## Image augmentation to prevent overfitting, our data set is somewhat small 
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory('../dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=batch,
                                                    class_mode='binary')
    
    test_set = test_datagen.flow_from_directory('../dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=batch,
                                                class_mode='binary')
    
    classifier.fit_generator(training_set,
                            steps_per_epoch=(8000/batch),
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=(2000/batch))
    
    elapsed = time.time() - start
    print("Elapsed: %f seconds", elapsed)
    
    from keras.models import load_model 
    classifier.save('cat_or_dog_220batch.h5')
    
    from keras.preprocessing import image
    import numpy as np
    prediction_image = image.load_img('./dataset/single_prediction/cat_or_dog_3.jpg')
    prediction_image = image.image_to_array(prediction_image)
    prediction_image = np.expand_dims(prediction_image, axis=0) 
    
    prediction_image = prediction_image.reshape(64, 64, 3)
    
    
    print(new_prediction)