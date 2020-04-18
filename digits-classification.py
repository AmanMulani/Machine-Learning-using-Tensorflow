import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd


def get_data(filename):
    with open(filename) as training_file:
        images = []
        labels = []
        training_file.readline()
        for row in training_file:
            row = row.split(',')
            label = np.array(row[0]).astype(np.float)
            image_string = np.array(row[1:785]).astype(np.float)
            
            image = np.array_split(image_string, 28)
            
            label = np.array(label) 
            image = np.array(image) 
            
            labels = np.append(labels, label)
            images.append(image)
            
    labels = np.array(labels).astype(float)
    images = np.array(images).astype(float)
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

training_images = np.expand_dims(training_images, axis = -1) 
testing_images = np.expand_dims(testing_images, axis = -1) 

train_datagen = ImageDataGenerator(
    # Your Code Here
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

validation_datagen = ImageDataGenerator(
    rescale = 1./255.
    )
    
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(25, activation = 'softmax')
])

model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])

history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size = 10),
                              validation_data = validation_datagen.flow(testing_images, testing_labels, batch_size = 10),
                              epochs = 25)

model.evaluate(testing_images, testing_labels, verbose=0)
