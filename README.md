
A basic neural network implementation in C++, in progress.

Uses [Armadillo](http://arma.sourceforge.net/) for linear algebra operations, [Catch v2.13.2](https://github.com/catchorg/Catch2) for testing.

This is a learning implementation, expanded from a homework which can be found [here](https://github.com/marifdemirtas/blg252e_2020/tree/main/oop_hw3).

## Command Line Usage

`./main (options_file) (train_x) (train_y) (output_directory) (test_x) (test_y)`

First argument is the path to options file for the network.
    First line contains one integer indicating number of layers (including input layer).
    Second line contains a list of integers, that indicates the neuron count in each layer (starting from input layer).
    Third line contains a list of integers, the activation of the neurons in corresponding layer.
        0 for Sigmoid
        1 for Leaky ReLU
        2 for ReLU

Second argument is the path to the x values of training samples.
    Each row contains one sample, each column corresponds to a feature.

Third argument is the path to the y values of training samples.
    Each row contains one sample.

Fourth argument is the path to an existing directory. If given, program will save the following under that directory.
    output.txt - The activation values of last layer
    weights.txt - Binary file for weights
    weight-i.txt - Human readable file for each weight
    biases.txt - Binary file for biases
    bias-i.txt - Human readable file for each bias

Fifth argument is the path to the x values of testing samples.
    Each row contains one sample, each column corresponds to a feature.

Sixth argument is the path to the y values of training samples.
    Each row contains one sample.

## To-do

- Implement softmax layer for multi-class classification
- Implement more methods to export/import models
- Complete tests for NetworkModel class