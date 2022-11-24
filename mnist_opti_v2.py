import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid

import matplotlib.pyplot as plt

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from nn_utils_function import *
from autils import *

np.set_printoptions(precision=2)

from mnist import MNIST
mndata = MNIST('./dir_with_mnist_data_files')
mndata.gz = True
images, labels = mndata.load_training()
images,labels = np.array(images)/255,np.array([labels]).T


print(images.shape)
print(labels.shape)

images_test, labels_test = mndata.load_testing()
images_test,labels_test = np.array(images_test)/255,np.array([labels_test]).T



#Step 3 : Optimization with different neurons
tf.random.set_seed(1234)

L1_neurons = [10,10,15,25,40,60,100,150,300]
L2_neurons = [5,8,10,10,15,20,40,60,120]
E_train = []
E_cv = []
for i in range(len(L1_neurons)):
    model = Sequential([
        tf.keras.Input(shape=(784,)),
        Dense(L1_neurons[i], activation='relu'),
        Dense(L2_neurons[i],activation='relu'),
        Dense(10,activation='linear')

    ],name='mnist_nn_v1')



    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )

    history = model.fit(
        images,labels,
        epochs=60
    )

    _,_,_,E_train = calculate_errors(model,images,labels,E_train)
    _,_,_,E_cv = calculate_errors(model,images_test,labels_test,E_cv)

n = [[L1_neurons[i]+L2_neurons[i]] for i in range(len(L1_neurons))]

#Step 2 : Optimization

plt.plot(n,E_train)
plt.plot(n,E_cv)
plt.legend(['E_train', 'E_cv'])
plt.xlabel('Neurons')
plt.ylabel('Error (%)')
plt.xscale('log')
plt.show()
