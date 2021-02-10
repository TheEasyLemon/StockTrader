"""
Tutorial Found here
https://www.tensorflow.org/tutorials/keras/classification

In this tutorial, we train a neural network model to classify images
of clothing. We'll use tf.keras, a high-level API to train models
in Tensorflow. ML_Workshop currently uses the same API.

Generally, when we say "training" an ANN (artificial neural network),
we mean that we are minimizing a loss, or cost, function. Of course, in
order to know how "wrong" our model is, we need correctly labelled data.
In this tutorial, Keras provides both - a dataset that contains training
data, and another for testing. In order to minimize our loss function,
we have to change the weights/biases of each perceptron/neuron and see if
the function goes down.

So for this example, our inputs will be the 28x28 pixel images of
clothing. Each pixel will have some greyscale value, which will
be fed into a neuron which has some internal weight attached to it.
It will also pass the result through a function and add a bias to it.
It will output the result to some middle layer, where there may
be multiple inputs to a given perceptron. That perceptron takes
the linear combination of its inputs based on its own weights
and then passes it through some function like tanh, the sigmoid function,
or ReLU (which is just max(x, 0)). The last layer should give us
some number that represents the classification, or what clothing item
the computer thinks the image represents.

When we start training, we initialize our weights randomly. The actual
process of minimizing the loss function can be complicated, here's
an overview of some common ones:

Stochastic Gradient Descent


The nice thing about the models we're dealing with is that the actual
model is able to be written out as an equation, and that equation
is differentiable. It may be extremely long, and the input space
will be hundreds of dimensions, but we can do math on it! We can
calculate the gradient of this equation, and it gives us some new
function that tells us the direction each variable should move for
a given set of weights. However, it doesn't tell us by how much
we should move - that's controlled by something called a hyper-parameter,
or the learning rate. You can think of this as the "dx", or differential.

So for each training step, for each weight, we subtract the derivative
with respect to that weight of the loss function multiplied by the
learning rate.

This can be very slow. People have come up with something called
mini-batch gradient descent - we approximate the derivate with a small
sample of the input space. More noisy learning, but faster. The size
of the batch is another hyper-parameter. A batch size of 1 is caled
Stochastic Gradient Descent. It has a tendency to better generalize
to inputs not in the training input.

Another note - when we talk about finding the gradient, it would be
insane to actually analytically find it. I can't believe I actually
thought we could...but a much easier way to is to estimate like so:

lr = learning rate
df/dx = (f(x1, x2, xi + lr, xn) - f(x1, x2, xi, xn)) / lr
for every i in weight.

Backpropagation


Another concept that is good to know is backpropagation. It's
another way to calculate the gradient using linear algebra. Precisely,
we need to find delC/delw and delC/delb for every weight w and bias b.
(C is the cost function).

To do this, we first start with a general definition of the outputs
of a given layer:

a^l = sigmoid(w^l * a^(l-1) + b^l)

a^l is simply the column vector of the outputs of a given layer.
w^l is a matrix of the weights, where each row symbolizes a given
neuron's input weights, and each column symbolizes some previous neuron's
weights for each of the current neurons.
b^l is a column vector of the biases of the outputs of a given layer.

We call the total input to the sigmoid function (w^l * a^(l-1) + b^l)
z^l, or the weighted input.

I could keep going, but this is already getting long. The rest is here:
http://neuralnetworksanddeeplearning.com/chap2.html

Adam

Adam is an adaptive learning rate optimization algorithm. The math is
complicated. If you want to know more, here it is:
https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c

Onto the actual Keras example!
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import and load fashion MNIST data
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# direct map of labels number to clothing article
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# data processing - convert 0 - 255 greyscale to 0.0 - 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

def show_training_data():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

"""
Building the model

The basic building block of the model is the layer. Layers are columns of neurons, I think. Not sure.
The nice thing about layers is that keras provides a lot of them....like, a lot.
https://www.tensorflow.org/api_docs/python/tf/keras/layers

Here are the ones we'll use:

Flatten - converts 2D array (image) to 1D - just a reformat layer.
Dense - fully connected layers - every single neuron is connected to every single other one
        in the next layer.

Then, we'll compile the model, giving it an optimizer (think Stochastic Gradient Descent, Adam, etc.), 
a loss function, and a metric. Learn more here:
https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile 
"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10) # output layer, we are categorizing 0-9 remember?
                              # also note how it doesn't give an integer output, but rather relative
                              # relative probabilities of it being 0-9, or any kind of clothing.
])

# compile that model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# We now train the model!!
model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# We now create a new model, which is just our old one with a softmax layer that
# converts the logits (linear outputs) to probabilities.
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# just as a proof of concept, we print the first prediction: the relative probability
# of the image any of the inputs given its training data:


def print_probabilities():
    print("Predictions for the first image: ")
    for i, prob in enumerate(predictions[0]):
        print(f"Probability of being a {class_names[i]}: {prob}\n")


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
