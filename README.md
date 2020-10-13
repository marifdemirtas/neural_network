make

./main tests/input6.txt tests/set6.txt (output_directory)

First argument is the settings for neural networks.
    First line contains one integer indicating number of layers (including input layer).
    Second line contains a list of integers, that indicates the neuron count in each layer (starting from input layer).
    Third line contains a list of integers, the activation of the neurons in corresponding layer.
        0 for Sigmoid
        1 for Leaky ReLU
        2 for ReLU

Second argument is the x values of examples.
    Each row contains one example, each column corresponds to a feature.

Third argument (optional) is path to an existing directory. If given, program will save the following under that directory.
    output.txt - The activation values of last layer
    weights.txt - Binary file for weights
    weight-i.txt - Human readable file for each weight
    biases.txt - Binary file for biases
    bias-i.txt - Human readable file for each bias