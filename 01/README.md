# Single layer Perceptron linear classifier - 2 dimensions

First assignment of the subject: ESE410020 - Special Topics in Electronics Systems I:

Develop a single layer Perceptron linear classifier - 2 dimensions. The code must contain the following functions:

* Function to train the Perceptron;
* Function to classify non-trained data with the Perceptron;
* Generate training and test data;
* Plot the line adjustments during the training - weights of the Perceptron;
* Plot the data and the decision border - line adjusted.

The project must have at least a readme file in .txt format indicating how to execute the submitted work. If you prefer to send as a report, it must have a maximum of one page.

Send the code, training data and other files in a single file in .zip format.

Deadline for submission 01/12/2019 (23:59)

## Prerequisites

The code is developed in [Python](https://www.python.org/downloads/) with the following libraries:

* [Numpy](https://numpy.org/) - math operations;
* [Matplotlib](https://matplotlib.org/) - plotting;
* [Random](https://docs.python.org/3/library/random.html) - generate data.

## Running

Running the 00.py file, a figure will open with the random generated data and the line adjustments will be plotted in a set of 100 iterations. Due to the aleatory data, if possible, it will be linearly separable. This shows that the single-layer Perceptron is working properly. Finished the training process, the weights will be printed. 

Therefore, you can click anywhere in the plot area and test the Perceptron with untrained points. It will print the selected coordinates and output, plotting it interactively with the correct classification. You can finish the program by closing the figure.

## Reference

* [John Sullivan](https://jtsulliv.github.io/perceptron/)
* [Thomas Countz](https://medium.com/@thomascountz/calculate-the-decision-boundary-of-a-single-perceptron-visualizing-linear-separability-c4d77099ef38)
* [Xu Liang](https://towardsdatascience.com/an-equation-to-code-machine-learning-project-walk-through-in-python-part-1-linear-separable-fd0e19ed2d7)

## Author

 **Mateus Abreu de Andrade** - [Linkedin](https://www.linkedin.com/in/mateus-abreu-de-andrade-92259659/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://opensource.org/licenses/MIT) file for details
