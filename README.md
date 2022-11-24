# Neural Net for MNIST dataset classification

In this project, we created a neural network with TensorFLow to classify the MNIST digit dataset.
This project was done with the help of the Coursera lecture on advanced learning algorythm : https://www.coursera.org/learn/advanced-learning-algorithms/home/welcome
<p align="center">
<img src='https://user-images.githubusercontent.com/96018383/203671856-26c9c4bf-9c84-4904-b18b-3fa31f71228e.png'>
</p>

## First Neural Network

We start with the neural net with 2 layers (with 25 and 15 neurons) with mnist_nn.py. We normalised the images values between 0 and 1.


We obtain the following loss curve during training :

<p align="center">
<img src='https://user-images.githubusercontent.com/96018383/203672654-94e62631-fec5-46f8-8477-f1803b1f12f1.png'>
</p>


With this neural net, we are able to classify 96.4% of the MNIST test set correctly.
Here is an exemple of missclassified images with the label guessed by the neural network :

<p align="center">
<img src='https://user-images.githubusercontent.com/96018383/203673017-d632de40-5be9-472c-b2ed-984311d88fd5.png'>
</p>


## Neural Net optimization

The next step optimize the neural net (with mnist_opti_v1.py). To do this, we start with the regularization parameter $\lambda$. We calculate the error on the training set and the test set as a function of $\lambda$ for varying between 0 and 0.3.

<p align="center">
<img src='https://user-images.githubusercontent.com/96018383/203718909-0a1fe270-3c47-4254-97c8-e5a256d17e3e.png'>
</p>



We can see that the error on the training set as well as on the testing set increase as $\lambda$ increase.
This mean that the initial NN, with $\lambda$=0 was not ovefitting the training data and give the best results.

Next we optimize the neural net by increasing the number of neurons (with mnist_opti_v2.py).
In the layer 1, the number of neurons varies from 10 to 1200 and, in the layer 2, the number of neurons varies from 5 to 480.

Here is the result we get : 

<p align="center">
<img src='https://user-images.githubusercontent.com/96018383/203688046-e595286e-9882-43e1-88c9-f4888e426910.png'>
</p>


We can see than the error on the training set as well as on the testing set decrease when we increase the number of neurones in the network.
The error on the testing set seems to reach the best value for 250 neurones and, since it does not increase afterward, our NN is not overfitting the data.

After optimization, we reach a best score of 98.3% of correct classification.

Here is an exemple of the misclassified data :

<p align="center">
<img src='https://user-images.githubusercontent.com/96018383/203694255-4bc712f7-e17f-4cda-816d-428fba84c884.png'>
</p>



To have better results, we can use a convolutional neural net (mnist_cnn.py).
With a CNN, we can see that the loss function on the testing data climb up after about 10 epochs which could mean we overfitted the data :

<p align="center">
<img src='https://user-images.githubusercontent.com/96018383/203719923-d56a5b6f-0263-480e-9eae-ce63acb5a404.png'>
</p>


With 10 epochs we were able to increase the success rate to 99.19% on the testing dataset.
Here is an exemple of the images not found. 

<p align="center">
<img src='https://user-images.githubusercontent.com/96018383/203720930-0d7248d1-edf2-458c-a2b2-93781f2b272e.png'>
</p>


Even for a human, some of this images are not easy to classify!

