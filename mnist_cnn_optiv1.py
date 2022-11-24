import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten
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

images = images.reshape(60000,28,28)

print(images.shape)
print(labels.shape)

images_test, labels_test = mndata.load_testing()
images_test,labels_test = np.array(images_test)/255,np.array([labels_test]).T
images_test = images_test.reshape(10000,28,28)
# plt.figure(1)
#plot_images(images*255,labels)


#Step 1 : NN
tf.random.set_seed(1234)
lambdas_ =[0,0.0001]#,0.001,0.01,0.1,0.3]

E_train = []
E_cv = []
for i in range(len(lambdas_)):
    model = Sequential([

        Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28,1),kernel_regularizer = tf.keras.regularizers.l2(lambdas_[i])),
        MaxPooling2D((2, 2)),
        Conv2D(64, (5, 5), activation='relu',kernel_regularizer = tf.keras.regularizers.l2(lambdas_[i])),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu',kernel_regularizer = tf.keras.regularizers.l2(lambdas_[i])),
        Flatten(),
        Dense(64, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(lambdas_[i])),
        Dense(32,activation='relu',kernel_regularizer = tf.keras.regularizers.l2(lambdas_[i])),
        Dense(10,activation='linear')

    ],name='mnist_nn_v1')

    #model.summary()

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )

    history = model.fit(
        images,labels,
        epochs=15,
        validation_data=(images_test, labels_test)
    )



    _,_,_,E_train = calculate_errors(model,images,labels,E_train,cnn=True)
    failed_X,failed_prediction,failed_y,E_cv = calculate_errors(model,images_test,labels_test,E_cv,cnn=True)

print(E_train)
plt.figure(4)
plt.plot(lambdas_,E_train)
plt.plot(lambdas_,E_cv)
plt.legend(['e_train', 'e_cv'])
plt.xlabel('lambda')
plt.ylabel('Error (%)')
plt.xscale('log')
plt.show()
