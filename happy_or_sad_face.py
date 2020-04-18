import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


'''
If you are developing in a local
environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
and place it inside a local folder and edit the path to that location
'''

path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

train_directory = os.path.join('/tmp/h-or-s')


def train_happy_sad_model():

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
         # Your Code
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('loss')<(1-DESIRED_ACCURACY)):
                self.model.stop_training = True

    callbacks = myCallback()
    
    model = tf.keras.models.Sequential([
        # Your Code Here
        tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (150,150, 3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150,150, 3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150,150, 3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr = 0.001), metrics = ['acc'])

    train_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size = (150, 150),
        batch_size = 8,
        class_mode = 'binary',
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 5,
        epochs = 15,
        validation_data = train_generator,
        validation_steps = 8,
        verbose = 2
    )

    return history.history['acc'][-1]


