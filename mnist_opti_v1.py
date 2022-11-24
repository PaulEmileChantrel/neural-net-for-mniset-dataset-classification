import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
#%matplotlib widget
import matplotlib.pyplot as plt

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from autils import *
from nn_utils_function import *
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


#plot_images(images,labels)


#Step 2 : Optimization by varying the regularisation parameter
tf.random.set_seed(1234)
lambdas_ =[0,0.001,0.003,0.01,0.05,0.1,0.2,0.3]

E_train = []
E_cv = []
for i in range(len(lambdas_)):
    model = Sequential([
        tf.keras.Input(shape=(784,)),
        Dense(25, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(lambdas_[i])),
        Dense(10,activation='relu',kernel_regularizer = tf.keras.regularizers.l2(lambdas_[i])),
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



print(E_train)
plt.figure(4)
plt.plot(lambdas_,E_train)
plt.plot(lambdas_,E_cv)
plt.legend(['e_train', 'e_cv'])
plt.xlabel('lambda')
plt.ylabel('Error (%)')
plt.xscale('log')
plt.show()
