
import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd


'''
This code block unzips the full Cats-v-Dogs dataset to /tmp
which will create a tmp/PetImages directory containing subdirectories
called 'Cat' and 'Dog' (that's how the original researchers structured it)
'''
path_cats_and_dogs = f"{getcwd()}/../tmp2/cats-and-dogs.zip"
shutil.rmtree('/tmp')

local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()


try:
    base_dir = os.path.join('/tmp/cats-v-dogs')
    
    os.mkdir(base_dir)
    
    training_dir = os.path.join('/tmp/cats-v-dogs/training')
    testing_dir = os.path.join('/tmp/cats-v-dogs/testing')
    
    os.mkdir(training_dir)
    os.mkdir(testing_dir)
    
    training_cats = os.path.join('/tmp/cats-v-dogs/training/cats')
    training_dogs = os.path.join('/tmp/cats-v-dogs/training/dogs')
    testing_cats = os.path.join('/tmp/cats-v-dogs/testing/cats')
    testing_dogs = os.path.join('/tmp/cats-v-dogs/testing/dogs')
    
    os.mkdir(training_cats)
    os.mkdir(training_dogs)
    os.mkdir(testing_cats)
    os.mkdir(testing_dogs)
    
except OSError:
    pass

    images = os.listdir(SOURCE)
    images = [image for image in images if os.path.getsize('{0}/{1}'.format(SOURCE, image))]
    images = random.sample(images, len(images))
    for image in images[:int(len(images)*SPLIT_SIZE)]:
        copyfile('{0}/{1}'.format(SOURCE, image), '{0}/{1}'.format(TRAINING, image))
        images.remove(image)
        
    for image in images:
        copyfile('{0}/{1}'.format(SOURCE, image), '{0}/{1}'.format(TESTING, image))


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])


model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])



TRAINING_DIR = '/tmp/cats-v-dogs/training'#YOUR CODE HERE
train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')#YOUR CODE HERE

train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size = 10, class_mode = 'binary', target_size = (150,150))#YOUR CODE HERE

VALIDATION_DIR = '/tmp/cats-v-dogs/testing' #YOUR CODE HERE
validation_datagen = ImageDataGenerator(rescale = 1./255.)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size = 10, class_mode = 'binary', target_size = (150, 150))#YOUR CODE HERE




history = model.fit_generator(train_generator,
                              epochs=2,
                              verbose=1,
                              validation_data=validation_generator,)

