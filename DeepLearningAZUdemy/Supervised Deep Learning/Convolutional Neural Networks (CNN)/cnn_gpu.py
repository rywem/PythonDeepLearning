# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import time
start = time.time()
# Part 1 - Building the Convolutional Neural Network
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

batch = 200
loadNetwork = True
saveNetwork = False
networkFileName = 'cat_or_dog_220batch.h5'

image_width = 128
image_height = 128


with tf.device('/gpu:0'):
    # Initialize the CNN
    classifier = Sequential()
    # Add convolutional layer
    classifier.add(Convolution2D(32, (3, 3), input_shape = (image_width, image_height, 3), activation = 'relu'))    
    # Max Pooling
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
        
    from keras.models import load_model 
    from keras.preprocessing.image import ImageDataGenerator
    ## Image augmentation to prevent overfitting, our data set is somewhat small     
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
  
    training_set = train_datagen.flow_from_directory('../dataset/training_set',
                                                    target_size=(image_width, image_height),
                                                    batch_size=batch,
                                                    class_mode='binary')
    
    test_set = test_datagen.flow_from_directory('../dataset/test_set',
                                                target_size=(image_width, image_height),
                                                batch_size=batch,
                                                class_mode='binary')  
    
    if loadNetwork == True:
        classifier = load_model(networkFileName)
        print("Loaded: ", networkFileName)
    else:    

        classifier.fit_generator(training_set,
                                steps_per_epoch=(8000/batch),
                                epochs=25,
                                validation_data=test_set,
                                validation_steps=(2000/batch))

        elapsed = time.time() - start
        print("Elapsed seconds: ", elapsed)
        
        if saveNetwork == True:
            classifier.save(networkFileName)        

    from keras.preprocessing import image
    import numpy as np  
    
    def loadImage(filename):        
        img = image.load_img(filename, target_size = (image_width, image_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0) # Adds an additional dimension to our test image, corresponds to the batch
        return img
    
    test_image = loadImage('./dataset/single_prediction/cat_or_dog_3.jpg')
    prediction = classifier.predict(test_image)
    indices = training_set.class_indices # Get corresponding category mapping
    
    if result[0][0] == 1:
        name_prediction = 'Dog'
    else:
        name_predicion = 'Cat'