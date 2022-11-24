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
E_train = []
E_cv = []

model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32,activation='relu'),
    Dense(10,activation='linear')

],name='mnist_nn_v1')

#model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    images,labels,
    epochs=10,
    validation_data=(images_test, labels_test)
)

#plt.figure(0)
plot_loss_tf(history)

_,_,_,E_train = calculate_errors(model,images,labels,E_train,cnn=True)
failed_X,failed_prediction,failed_y,E_cv = calculate_errors(model,images_test,labels_test,E_cv,cnn=True)

plt.figure(1)
plot_images(failed_X.reshape(failed_X.shape[0],784)*255,failed_prediction)

plt.show()
