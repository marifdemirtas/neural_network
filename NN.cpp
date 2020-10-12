/*
Student Name: Mehmet Arif Demirtaş
Student ID : 150180001
*/

#include <iostream>
#include <armadillo>
#include <cassert>
#include "NN.h"

void SigmoidLayer::activate()
{
    a_vals = 1 / (1 + arma::exp(-z_vals));
};

void ReluLayer::activate()
{
    a_vals = arma::max(arma::zeros(z_vals.n_rows, z_vals.n_cols), z_vals);
};

void LReluLayer::activate()
{
    a_vals = arma::max(z_vals * 0.1, z_vals);
};

Layer::Layer(int neuron_count):a_vals(neuron_count, 1, arma::fill::ones), z_vals(neuron_count, 1, arma::fill::ones)
{
    this->neuron_count = neuron_count;
}

void Layer::setValues(double* z_vals)
{
    for (int i = 0; i < neuron_count; ++i){
        this->z_vals(i) = z_vals[i];
    }
    this->a_vals = this->z_vals;
}

void Layer::showActiveValues(){
//    for(int i = 0; i < neuron_count; i++){
//        std::cout << neurons[i].getA() << std::endl;
//    }
    std::cout << a_vals << std::endl;    
}

void Layer::computeZVals(arma::mat weight, arma::vec bias, arma::mat a_vals){
    /*
    DOUBLE* TORETURN = NEW DOUBLE[NEXT_LAYER_SIZE]{0}
    FROM 1 TO NEXT_LAYER_SIZE
        FROM 1 TO THIS_LAYER_SIZE
            TORETURN[i] += WEIGHT[i][j] * THIS[j]
        TORETURN[i] += BIAS[i]
    RETURN TORETURN
    */
    assert(weight.n_rows == z_vals.n_rows);
    assert(weight.n_cols == a_vals.n_rows);
    assert(bias.n_rows == weight.n_rows);

    this->z_vals = weight * a_vals;
    this->z_vals.each_col() += bias;
}

Network::Network(int layer_count, int* neuron_counts, int* neuron_types):weights(layer_count), biases(layer_count)
{
    this->layer_count = layer_count;
    this->neuron_counts = neuron_counts;

    layers = new Layer*[layer_count];

    for (int i = 0; i < layer_count; ++i){
        switch(neuron_types[i]){
            case 0:
                layers[i] = new SigmoidLayer(neuron_counts[i]);
                break;
            case 1:
                layers[i] = new LReluLayer(neuron_counts[i]);
                break;
            case 2:
                layers[i] = new ReluLayer(neuron_counts[i]);
                break;
            default:
                throw std::string("Unidentified activation function!");
        }
        weights(i) = arma::randn(neuron_counts[i], neuron_counts[i-1]);
        weights(i).fill(0.1);
        biases(i) = arma::randn(neuron_counts[i]);
        biases(i).fill(0.1);
    }

/*
    weights = new double**[layer_count - 1]; //Array of 2D arrays
                                             //weights[i] is the weight matrix associated with the transition from layer i to i+1 
    for (int i = 0; i < layer_count - 1; i++){
        weights[i] = new double*[neuron_counts[i + 1]]; //weights[i] has row count equal to the number of neurons in the layer i+i
                                                        //since in the matrix multiplication Z_i+1 = W_i * A_i, result should be in shape 1 x COLUMN_i+1
        for (int j = 0; j < neuron_counts[i + 1]; j++){
            weights[i][j] = new double[neuron_counts[i]]; //weights[i][j] is the row vector with which neurons of layer i are multiplied to obtain a neuron in layer i+1
            for (int k = 0; k < neuron_counts[i]; ++k){
                weights[i][j][k] = 0.1;                                     //SET INITIAL WEIGHT TO 0.1
            }
        }
    }
*/

/*
    biases = new double*[layer_count - 1]; //Array of arrays, biases[i] is the biases for transition from layer i to i+1
    for (int i = 0; i < layer_count - 1; i++){
        biases[i] = new double[neuron_counts[i+1]]; //biases[i] contains an entry for each neuron in layer i+1
        for (int j = 0; j < neuron_counts[i+1]; j++){
            biases[i][j] = 0.1;                                             //SET INITIAL BIAS TO 0.1
        }
    }
*/
}

void Network::forwardPropagate(double* input_vals){
    layers[0]->setValues(input_vals);
    for (int i = 1; i < layer_count; i++){
        layers[i]->computeZVals(weights(i), biases(i), layers[i-1]->getA());
        layers[i]->activate();
    }
}

void Network::showActiveValues(){
    for (int i = 0; i < layer_count; i++){
        std::cout << "Layer " << i << ":" << std::endl;
        layers[i]->showActiveValues();
    }
}

Network::~Network(){
    for (int i = 0; i < layer_count; ++i)
    {
        delete layers[i];
    }
    delete[] layers;

/*
    for (int i = 0; i < layer_count - 1; i++){
        for (int j = 0; j < neuron_counts[i + 1]; j++)
        {
            delete[] weights[i][j];
        }
        delete[] weights[i];
    }
    delete[] weights;

    for (int i = 0; i < layer_count; i++)
    {
        delete[] biases[i];
    }
    delete[] biases;
*/
    delete[] neuron_counts;
}