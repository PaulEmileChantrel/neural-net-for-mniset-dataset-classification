
import matplotlib.pyplot as plt
import numpy as np


def plot_images(X,y)->None:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = X.shape

    fig, axes = plt.subplots(4,4, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]


    for i,ax in enumerate(axes.flat):
        # Select random indices

        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((28,28))

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Display the label above the image
        ax.set_title(y[random_index,0])
        ax.set_axis_off()
        fig.suptitle("Label, image", fontsize=14)
    #plt.show()


def plot_loss_tf(history):
    losses = history.history['loss']
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('losses')

def calculate_errors(model,X,y,E):
    failed_X = []
    failed_y = []

    test_size = X.shape[0]
    prediction = model.predict(X.reshape(test_size,784))
    prediction = np.argmax(prediction,axis=1)

    failed_X = [X[i,:] for i in range(len(prediction)) if prediction[i]!=y[i]]
    failed_prediction = [prediction[i] for i in range(len(prediction)) if prediction[i]!=y[i]]
    failed_y = [y[i] for i in range(len(prediction)) if prediction[i]!=y[i]]
    percent_failed = round(len(failed_X)/test_size*100*100)/100
    print(f'{percent_failed} % of failure (or {len(failed_X)} out of {test_size})')

    #append to E_cv or E_train
    E.append(percent_failed)

    return np.array(failed_X),np.array([failed_prediction]).T,np.array([failed_y]).T,E
