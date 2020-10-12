make

./main tests/input6.txt tests/set6.txt

First argument is the settings for neural networks.
    First line contains one integer indicating number of layers (including input layer).
    Second line contains a list of integers, that indicates the neuron count in each layer (starting from input layer).
    Third line contains a list of integers, the activation of the neurons in corresponding layer.
        0 for Sigmoid
        1 for Leaky ReLU
        2 for ReLU

Second argument is the x values of examples.
    Each row contains one example, each column corresponds to a feature.