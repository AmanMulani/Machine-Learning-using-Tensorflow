'''We are trying a simple linear regression program using tensorflow. 
Here instead of using linear regression, we are using the most simple neural network with just one neuron.
'''
import tensorflow as tf
import numpy as np
from tensorflow import keras


'''
First we will make a model, i.e. a neural network, here it's a Sequential Model from keras.
units parameter tells the number of neurons and input share specifies the dimensions of the input.
'''
model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

'''
Here's where tensorflow is the most useful, you don't need to hard code the optimizers and loss functions.
'''
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

'''
xs is the input and ys is the output.
We will train our model on these parameters and then try to find out the output ys, when xs is provided as an input.
'''
xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ys = np.array([-1, 1, 3, 5, 7, 9, 11, 13, 15, 17])

#Here we train the model.
model.fit(xs, ys, epochs = 500)

# xs = 30 is given as an input and the corresponding value of ys is printed.
print(model.predict([30]))

