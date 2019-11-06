# Multilayer Perceptron number classifier - Back propagation

Second assignment of the subject: ESE410020 - Special Topics in Electronics Systems I:

Develop a MLP number classifier - 0 to 9. The code must contain the following functions:

* Do not use any MLP library - build code from scratch;
* Function train the MLP with the following characteristics:
* Inputs: input data, output data,number of layers, number of neurons per layer, learning rate, momentum and number of max epochs;
* Outputs: mean square error of each epoch and the weights;
* Function test the MLP with the following characteristics:
* Inputs: untrained data;
* Outputs: output of the MLP;
* Data for training and test is provided - 8x8 matrix with 64 positions that represents each number;
* You should add new data to train and test the MLP - like noise in the original data;
* Try different combinations of inputs in the training function;
* Plot model convergence and generalization;
* Test different generalized estimation methods: hold-out, cross-validation, bootstrap;
* 3 pages report: 
* Critical analysis and comparison between the different methods;
* Graphs and tables discussing the results.

The project must have at least a readme file in .txt format indicating how to execute the submitted work and the report.

Send all the files in a single .zip file.

Deadline for submission 01/12/2019 (23:59)

## Prerequisites

The code is developed in [Python](https://www.python.org/downloads/) with the following libraries:

* [Numpy](https://numpy.org/) - math operations;
* [Matplotlib](https://matplotlib.org/) - plotting;
* [Random](https://docs.python.org/3/library/random.html) - generate random data.
* [Tkinter](https://docs.python.org/3/library/tkinter.html) - Graphical User Interface

## Running

First, define the neural network number of layers, neurons and activation function (sigmoid or tanh)by the method .add_layer in the # Create Neural Network block - The number of neurons must be the same size of the previous layer. Remember that,the data provided are 64 inputs and 10 outputs. Therefore, in the # Train parameters block, define the learning rate, momentum, number of epochs and the type of generalized estimation methods: None,hold-out, k-fold or bootstrap. If desired, you can use a decay function in the learning rate and momentum variables by setting the function and rate - time-based or drop-based in every 50 epochs.

By setting the type parameter to None, all the data will be used to train the defined neural network with the number of epochs defined. Completing that,four graphs will be plotted with the changes in mean square error (MSE), accuracy, learning rate and momentum during the training. Also,the weights in each layer of the neural network will be printed in the terminal. After closing this figure, a interface with 64 buttons that represent the 8x8 matrix digit,will appear and as you click in the buttons, they value will turn from 0 to 1, white to black and vice versa. Moreover,there are two additional buttons and labels. By pressing the Apply Input button, the label after the Predicted Output will display the result of the neural network for this input. Although,if you press the Clear matrix button, all the digits buttons will turn to zero and no input will be applied. To finish the script, just finish this window.

However, if you set the type parameter to hold-out the defined train percentage amount will be used to train the neural network and the remain data to test. The times parameter define the amount of completing training's will be performed delimited by the number of epochs shuffling the data input and new random weights values in every new train. After that, four graphs will be plotted with the changes in mean square error (MSE), accuracy,learning rate and momentum during the training the process applying the test data in the neural network. As a second option,you can set the type parameter to k-fold in order to perform the k-fold cross-validation generalized estimation method. This time,the times parameter is used to define how many equal-sized parts the data will be divided and consequently the number of training's. The same graph will be plotted.

Finally, the parameter type can be set to bootstrap, which the times parameter will define the amount of training's applying the method. Also,the graphs will be plotted with the same performance variables and neural network characteristics.

## Reference

* [Zanid Haytam](https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/)
* [Jason Brownlee](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)
* [Samay Shamdasani](https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python)
* [Sagar Sharma](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

## Author

 **Mateus Abreu de Andrade** - [Linkedin](https://www.linkedin.com/in/mateus-abreu-de-andrade-92259659/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://opensource.org/licenses/MIT) file for details
