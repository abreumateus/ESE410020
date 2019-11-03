# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tkinter import *
import os

# Represents a layer (hidden or output) in our neural network.
class Layer:

    def __init__(self, n_input, n_neurons, activation=None):

        # int n_input: The input size (coming from the input layer or a previous hidden layer)
        # int n_neurons: The number of neurons in this layer.
        # weights: The layer's weights.
        # str activation: The activation function to use (if any).
        # bias: The layer's bias.
        # last_activation: The last activated output of the neuron.
        # error: Error between the desired output and the activated output.
        # delta: Weights delta.
        # last_delta: The last delta of the layer.
        
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.weights = np.random.randn(n_input, n_neurons)
        self.activation = activation
        self.bias = np.random.randn(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None
        self.last_delta = 0

    #Calculates the dot product of this layer.
    def activate(self, x):
        
        # x: The input.
        # return: The activated output.

        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    # Applies the chosen activation function (if any).
    def _apply_activation(self, r):
        
        # r: The normal value.
        # return: The "activated" value.

        # sigmoid.
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        # tanh.
        if self.activation == 'tanh':
            return (np.exp(r) - np.exp(-r)) / (np.exp(r) + np.exp(-r))

    # Applies the derivative of the activation function (if any).
    def apply_activation_derivative(self, r):

        # r: The normal value.
        # return: The "derived" value.
        # 'r' is directly used here because its already activated, the only values that are used in this function are the last activations that were saved.

        # sigmoid.
        if self.activation == 'sigmoid':
            return r * (1 - r)

        # tanh.
        if self.activation == 'tanh':
            return (1 - np.power(r,2))

# Represents a neural network.
class NeuralNetwork:
    
    # Initialize the vectors of layers.
    def __init__(self):
        self._layers = []
    
    # Add a layer to the neural network.
    def add_layer(self, layer):

        # Layer layer: The layer to add.

        self._layers.append(layer)

    # Feed forward the input through the layers.
    def feed_forward(self, X):
    
        # X: The input values.
        # return: The output of the last layer.

        for layer in self._layers:
            X = layer.activate(X)

        return X

    # Performs the backward propagation algorithm and updates the layers weights.
    def backpropagation(self, X, y, learning_rate, momentum):
        
        # X: The input values.
        # y: The target values.
        # float learning_rate: The learning rate (between 0 and 1).
        # float momentum: The momentum rate (between 0 and 1).

        # Feed forward for the output.
        output = self.feed_forward(X)

        # Loop over the layers backward.
        for layer in reversed(self._layers):
            # If this is the output layer.
            if layer == self._layers[-1]:
                layer.error = y - output
                # The output = layer.last_activation in this case.
                layer.delta = layer.error * layer.apply_activation_derivative(output) - momentum * layer.last_delta
                layer.last_delta = layer.delta
            else:
                layer.error = np.dot(next_layer.delta, next_layer.weights.T)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation) - momentum * layer.last_delta
                layer.last_delta = layer.delta
            next_layer = layer
        # Update the weights.
        for layer in self._layers:
            # The input is either the previous layers output or X itself (for the first hidden layer).
            if layer == self._layers[0]:
                layer.weights += learning_rate * np.dot(X.T, layer.delta)
            else:
                layer.weights += learning_rate * np.dot(layer_before.last_activation.T, layer.delta)
            layer_before = layer

    # Trains the neural network using backpropagation.
    def train(self, data, max_epochs, learning_rate, l_decay_rate, l_decay_type, momentum, m_decay_rate, m_decay_type, type, train_percentage=None, times=None):
        
        # data: The input values and target values for training.
        # int max_epochs: The maximum number of epochs (cycles).
        # float learning_rate: The learning rate (between 0 and 1).
        # float l_decay_rate: The decay rate (between 0 and 1).
        # float l_decay_type: None, time-based or drop-based.
        # float momentum: The momentum (between 0 and 1).
        # float m_decay_rate: The decay rate (between 0 and 1).
        # float m_decay_type: None, time-based or drop-based.
        # type: None, hold-out, cross-validation or bootstrap.
        # return: The list of calculated MSE errors, learning_rate, momentum and accuracy.

        lrs = []
        mmts = []

        # Train with all the data.
        if type == None:
            #Variables
            mses = []
            accs = []
            lrs.append(learning_rate)
            mmts.append(momentum)
            # Number of epochs.
            for i in range(max_epochs):
                # Shuffle data input inevery epoch.
                np.random.shuffle(data)
                X = np.vstack(data[:,10:74])
                y = np.vstack(data[:,0:10])
                # Save the learning rate, momentum, accuracy and mean square error in a vector.
                lrs.append(self.decay(learning_rate,lrs[i],l_decay_rate,i,max_epochs,l_decay_type))
                mmts.append(self.decay(momentum,mmts[i],m_decay_rate,i,max_epochs,m_decay_type))
                accs.append(self.accuracy(nn.feed_forward(X), y)*100)
                self.backpropagation(X, y, lrs[i], mmts[i])
                mse = np.mean(np.square(y - nn.feed_forward(X)))
                mses.append(mse)
            # Return data.
            return mses, lrs, mmts, accs
        
        # Train with hold-out evaluation method.
        if type == 'hold-out':
            # Create matrix to save data.
            mses = [[0 for x in range(max_epochs)] for y in range(times)]
            accs = [[0 for x in range(max_epochs)] for y in range(times)]
            lrs.append(learning_rate)
            mmts.append(momentum)
            # Select the defined training percentage to train.
            training_range = int(train_percentage*np.shape(data)[0])
            # Number of epochs.
            for j in range(times):
                # Shuffle data input every epoch.
                np.random.shuffle(data)
                data_0 = data[0:training_range,:]
                data_1 = data[training_range:np.shape(data)[0],:]
                self.reset()
                for i in range(max_epochs):
                    # Shuffle data input every epoch.
                    np.random.shuffle(data_0)
                    X_train = np.vstack(data_0[:,10:74])
                    X_test = np.vstack(data_1[:,10:74])
                    y_train = np.vstack(data_0[:,0:10])
                    y_test = np.vstack(data_1[:,0:10])
                    # Save the learning rate, momentum, accuracy and mean square error in a vector.
                    lrs.append(self.decay(learning_rate,lrs[i],l_decay_rate,i,max_epochs,l_decay_type))
                    mmts.append(self.decay(momentum,mmts[i],m_decay_rate,i,max_epochs,m_decay_type))
                    accs[j][i] = self.accuracy(nn.feed_forward(X_test), y_test) * 100
                    self.backpropagation(X_train, y_train, lrs[i], mmts[i])
                    mse = np.mean(np.square(y_test - nn.feed_forward(X_test)))
                    mses[j][i] = mse
            # Return data.
            return mses, lrs, mmts, accs
        
        # Train with k-fold evaluation method.
        if type == 'k-fold':
            # Create matrix to save data.
            mses = [[0 for x in range(max_epochs)] for y in range(times)]
            accs = [[0 for x in range(max_epochs)] for y in range(times)]
            lrs.append(learning_rate)
            mmts.append(momentum)
            divided_data = np.array_split(data,times)
            # Number of loops.
            for j in range(times):
                # Delete one of the k-parts of the data.
                data_0 = np.delete(divided_data,j,axis=0)
                data_0 = np.resize(data_0,((np.shape(data)[0]-np.shape(divided_data)[1]),np.shape(data)[1]))
                data_1 = divided_data[j][0:np.shape(divided_data)[1]]
                self.reset()
                # Number of epochs.
                for i in range(max_epochs):
                    # Shuffle data input every epoch.
                    np.random.shuffle(data_0)
                    X_train = np.vstack(data_0[:,1:65])
                    X_test = np.vstack(data_1[:,1:65])
                    y_train = np.vstack(data_0[:,0])
                    y_test = np.vstack(data_1[:,0])
                    # Save the learning rate, momentum, accuracy and mean square error in a vector.
                    lrs.append(self.decay(learning_rate,lrs[i],l_decay_rate,i,max_epochs,l_decay_type))
                    mmts.append(self.decay(momentum,mmts[i],m_decay_rate,i,max_epochs,m_decay_type))
                    accs[j][i] = self.accuracy(nn.feed_forward(X_test), y_test) * 100
                    self.backpropagation(X_train, y_train, lrs[i], mmts[i])
                    mse = np.mean(np.square(y_test - nn.feed_forward(X_test)))
                    mses[j][i] = mse
            # Return data.
            return mses, lrs, mmts, accs
        
        # Train with bootstrap evaluation method.
        if type == 'bootstrap':
            # Create matrix to save data.
            mses = [[0 for x in range(max_epochs)] for y in range(times)]
            accs = [[0 for x in range(max_epochs)] for y in range(times)]
            lrs.append(learning_rate)
            mmts.append(momentum)
            data_size = (np.shape(data)[0])-1
            # Number of loops.
            for j in range(times):
                # Creat new random data in every epoch.
                data_0 =[]
                data_0 = np.hstack(data[random.randint(0,np.shape(data)[0]-1)])
                for i in range(data_size):
                    np.random.shuffle(data_0)
                    data_0 = np.vstack((data_0,data[random.randint(0,np.shape(data)[0]-1)]))
                self.reset()
                # Number of epochs.
                for i in range(max_epochs):
                    # Shuffle data input every epoch.
                    np.random.shuffle(data_0)
                    X_train = np.vstack(data_0[:,1:65])
                    X_test = np.vstack(data[:,1:65])
                    y_train = np.vstack(data_0[:,0])
                    y_test = np.vstack(data[:,0])
                    # Save the learning rate, momentum, accuracy and mean square error in a vector.
                    lrs.append(self.decay(learning_rate,lrs[i],l_decay_rate,i,max_epochs,l_decay_type))
                    mmts.append(self.decay(momentum,mmts[i],m_decay_rate,i,max_epochs,m_decay_type))
                    accs[j][i] = self.accuracy(nn.feed_forward(X_test), y_test) * 100
                    self.backpropagation(X_train, y_train, lrs[i], mmts[i])
                    mse = np.mean(np.square(y_test - nn.feed_forward(X_test)))
                    mses[j][i] = mse
            # Return data.
            return mses, lrs, mmts, accs


    # Calculate the accuracy between the predicted  output and the training output.
    def accuracy(self, y_pred, y_true):
       
        # y_pred: The predicted labels.
        # y_true: The true labels.
        # return: The calculated accuracy.

        return ((np.round(y_pred,1)== y_true)).mean()

    # Transforms the neurons outputs in numbers again.
    def translate(self, y_pred):

        # y_pred: The predicted labels.
        
        y_n = 0
        y_0 = []
        # Transform the float trained data into 0 and 1.
        for i in range(np.shape(y_pred)[0]):
            if(y_pred[i] >= 0.5):
                y_0.append(1)
            else:
                y_0.append(0)
        # Transform the output array into number from 0 to 9.
        for i in range(np.shape(y_pred)[0]):
            if(y_0[i] == 1):
                y_n = i
        return y_n

    # Reset the neural Networking training parameters.
    def reset(self):
        
        for layer in self._layers:
            layer.weights = np.random.randn(layer.n_input, layer.n_neurons)
            layer.bias = np.random.randn(layer.n_neurons)
            layer.last_activation = None
            layer.error = None
            layer.delta = None
            layer.last_delta = 0

    # Decay functions
    def decay(self, x_0, x, decay_rate, epoch, max_epochs, type):
    
    # x: Normal value.
    # decay_rate: The decay rate.
    # epoch: The actual epoch.
    # max_epochs: The total o epochs.
    # return: The calculated variable based on the decay type.
    # type: The type of decay

        if type == None:
            return x
        # Decay based on every epoch
        if type == 'time-based':
            return x / (1 + decay_rate/max_epochs * epoch)
        # Decay based on every 50 epochs
        if type == 'drop-based':
            return x_0 * math.pow((1-decay_rate), math.floor((1+epoch)/50))

# Create Neural Network.
np.random.seed(100)
nn = NeuralNetwork()
# def __init__(self, n_input, n_neurons, activation=None):
nn.add_layer(Layer(64, 15, 'sigmoid'))
nn.add_layer(Layer(15, 10, 'sigmoid'))
#nn.add_layer(Layer(10, 10, 'sigmoid'))
    
# Import data.
script_file_path  = os.path.dirname(os.path.abspath(__file__))
myFile = np.genfromtxt(script_file_path + '/data.csv', delimiter=',')

# Transforms the data ouput to fit the output layer size.
y_zeros = np.zeros((np.shape(myFile)[0], 10))
for i in range(np.shape(myFile)[0]):
    y_zeros[i][int(myFile[i][0])] = 1
myFile = np.delete(myFile,0,axis=1)
myFile = np.hstack((y_zeros,myFile))

# Train paremeters.
# def train(self, data, max_epochs, learning_rate, l_decay_rate, l_decay_type, momentum, m_decay_rate, m_decay_type, type, train_percentage=None, times=None)
learning_rate = 0.25 # 0 to 1
l_decay_rate = 0.001 # 0 to 1
l_decay_type = 'time-based' #None, time-base, drop-based
momentum = 0.5 # 0 to 1
m_decay_rate = 0.05 # 0 to 1
m_decay_type = 'drop-based' #None, time-base, drop-based
n_epochs = 1000
type = None # None, hold-out, k-fold, bootstrap
train_percentage = 0.7 # 0 to 1
times = 10

# Subplots: MSE, Learning rate, Momentum and Accuracy.
if type == None:
    
    # Train the neural network.
    mses, lrs, mmts, accs = nn.train(myFile, n_epochs, learning_rate,l_decay_rate,l_decay_type, momentum, m_decay_rate, m_decay_type, type)
    
    # Print layers weights.
    for i in range(np.shape(nn._layers)[0]):
        print('Weights layer ' + str(i))
        print(nn._layers[i].weights)
    
    # Plot the MSE, learning rate, momentum and accuracy.
    fig, axis = plt.subplots(2,2, figsize=(15,8))
    fig.canvas.set_window_title('MLP training results')

    # MSE.
    axis[0][0].plot(mses,'tab:blue')
    axis[0][0].set_title('Changes in MSE')
    axis[0][0].grid(True)
    axis[0][0].set(xlabel='Epochs', ylabel='MSE')
    
    # Accuracy.
    axis[0][1].plot(accs,'tab:orange')
    axis[0][1].set_title('Changes in Accuracy')
    axis[0][1].grid(True)
    axis[0][1].set(xlabel='Epochs', ylabel='Accuracy')
    
    # Learning rate.
    axis[1][0].plot(lrs,'tab:green')
    axis[1][0].set_title('Changes in Learning Rate')
    axis[1][0].grid(True)
    axis[1][0].set(xlabel='Epochs', ylabel='Learning Rate')
    
    # Momentum.
    axis[1][1].plot(mmts,'tab:red')
    axis[1][1].set_title('Changes in Momentum')
    axis[1][1].grid(True)
    axis[1][1].set(xlabel='Epochs', ylabel='Momentum')

    fig.tight_layout()
    plt.show()

    # Interface to apply test inputs.
    interface = Tk()
    matrix_size = 8

    # Variables.
    text = [[None]*matrix_size for _ in range(matrix_size)]
    buttons = [[None]*matrix_size for _ in range(matrix_size)]

    # GUI Title.
    interface.title("MLP number classifier - 0 to 9" )

    # Change color and number.
    def click(i,j):
        if text[i][j].get() == '0':
            text[i][j].set('1')
            buttons[i][j].config(bg='black')
        else:
            text[i][j].set('0')
            buttons[i][j].config(bg='white')

    # Apply input.
    def apply_input():
        input = []
        for i in range(matrix_size):
            for j in range(matrix_size):
                input.append(int(text[i][j].get()))
        label2.config(text = str(nn.translate(np.round(nn.feed_forward(input),1))))

    # Clear matrix.
    def clear_matrix():
        for i in range(matrix_size):
            for j in range(matrix_size):
                text[i][j].set('0')
                buttons[i][j].config(bg='white')

    # Matrix 8x8.
    for i in range(matrix_size):
        for j in range(matrix_size):
            text[i][j] = StringVar()
            text[i][j].set('0')
            buttons[i][j] = Button(interface, command = lambda i=i, j=j : click(i,j), bg = 'white')
            buttons[i][j].config(textvariable = text[i][j], width = 2, height = 2)
            buttons[i][j].grid(row = i, column = j)

    # Button apply input.
    bt01 = Button(interface, command = lambda : apply_input(), bg = 'snow3',text = 'Apply input')
    bt01.config(width = 20, height = 2)
    bt01.grid(row = 8, column = 0, columnspan = 4)

    # Button reset matrix.
    bt02 = Button(interface, command = lambda : clear_matrix(), bg = 'snow3',text = 'Clear matrix')
    bt02.config(width = 20, height = 2)
    bt02.grid(row = 8, column = 4, columnspan = 4)

    # Label output predicted.
    label1 = Label( interface, width = 23, height = 2, bg = 'snow2', text = 'Predicted Output:')
    label1.grid(row = 9, column = 0, columnspan = 4)

    # Label result ouput.
    label2 = Label( interface, width = 23, height = 2, bg = 'snow2', text = '0')
    label2.grid(row = 9, column = 4, columnspan = 4)

    interface.mainloop()
    
if type == ('hold-out') or type == ('k-fold') or type == ('bootstrap'):
        
    # Train the neural network.
    mses, lrs, mmts, accs = nn.train(myFile, n_epochs, learning_rate,l_decay_rate,l_decay_type, momentum, m_decay_rate,m_decay_type, type, train_percentage,times)

    # Plot the MSE, learning rate, momentum and accuracy.
    fig, axis = plt.subplots(2,2, figsize=(15,8))
    fig.canvas.set_window_title('MLP training results')

    # MSE.
    for i in range(times):
        axis[0][0].plot(mses[i], label = "MSE"+str(i))
    axis[0][0].set_title('Changes in MSE')
    axis[0][0].grid(True)
    axis[0][0].set(xlabel='Epochs', ylabel='MSE')
    axis[0][0].legend(loc="upper right")
    
    # Accuracy.
    for i in range(times):
        axis[0][1].plot(accs[i], label = "ACC"+str(i))
    axis[0][1].set_title('Changes in Accuracy')
    axis[0][1].grid(True)
    axis[0][1].set(xlabel='Epochs', ylabel='Accuracy')
    axis[0][1].legend(loc="upper right")

    # Learning rate.
    axis[1][0].plot(lrs,'tab:green')
    axis[1][0].set_title('Changes in Learning Rate')
    axis[1][0].grid(True)
    axis[1][0].set(xlabel='Epochs', ylabel='Learning Rate')

    # Momentum.
    axis[1][1].plot(mmts,'tab:red')
    axis[1][1].set_title('Changes in Momentum')
    axis[1][1].grid(True)
    axis[1][1].set(xlabel='Epochs', ylabel='Momentum')

    fig.tight_layout()
    plt.show()