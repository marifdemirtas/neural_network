/*
Student Name: Mehmet Arif Demirtaş
Student ID : 150180001
*/

#include <iostream>
#include <cmath>

#include "NN.h"

Neuron::Neuron(double z){
    this->z = z;
    this->a = z;
}

void SigmoidNeuron::activate(){
    setA(1 / (1 + exp(-getZ())));
}

void ReluNeuron::activate(){
    setA(std::max(0.0,getZ()));
}

void LReluNeuron::activate(){
    setA(std::max(getZ() / 10.0, getZ()));    
}

void Layer::init(int neuron_count, int neuron_type){
    switch(neuron_type){
        case 0:
            neurons = new SigmoidNeuron[neuron_count];
            break;
        case 1:
            neurons = new LReluNeuron[neuron_count];
            break;
        case 2:
            neurons = new ReluNeuron[neuron_count];
            break;
        default:
            throw std::string("Unidentified activation function!");
    }

    this->neuron_count = neuron_count;
}

Layer::~Layer(){
    delete[] neurons;
}

void Layer::setValues(double* z_vals){
    for (int i = 0; i < neuron_count; ++i)
    {
        neurons[i].setZ(z_vals[i]);
        neurons[i].setA(z_vals[i]);
    }
}

void Layer::activate(){
    for(int i = 0; i < neuron_count; i++){
        neurons[i].activate();
    }    
}

void Layer::showActiveValues(){
    for(int i = 0; i < neuron_count; i++){
        std::cout << neurons[i].getA() << std::endl;
    }    
}

double* Layer::computeNextLayer(double** weights, double* bias, int next_layer_size){
    /*
    DOUBLE* TORETURN = NEW DOUBLE[NEXT_LAYER_SIZE]{0}
    FROM 1 TO NEXT_LAYER_SIZE
        FROM 1 TO THIS_LAYER_SIZE
            TORETURN[i] += WEIGHT[i][j] * THIS[j]
        TORETURN[i] += BIAS[i]
    RETURN TORETURN
    */
    
    double* next_layer_vals = new double[next_layer_size];

    for (int i = 0; i < next_layer_size; i++){          //MATRIX OP EQ: Z_{i+1} = W_{i} * A_{i} + B_{i}
        for (int j = 0; j < this->getCount(); j++){
            next_layer_vals[i] += weights[i][j] * neurons[j].getA();
        }
        next_layer_vals[i] += bias[i];
    }

    return next_layer_vals;
}

Network::Network(int layer_count, int* neuron_counts, int* neuron_types){

    this->layer_count = layer_count;
    this->neuron_counts = neuron_counts;

    layers = new Layer[layer_count];
    for (int i = 0; i < layer_count; i++){
       layers[i].init(neuron_counts[i], neuron_types[i]);
    }

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

    biases = new double*[layer_count - 1]; //Array of arrays, biases[i] is the biases for transition from layer i to i+1
    for (int i = 0; i < layer_count - 1; i++){
        biases[i] = new double[neuron_counts[i+1]]; //biases[i] contains an entry for each neuron in layer i+1
        for (int j = 0; j < neuron_counts[i+1]; j++){
            biases[i][j] = 0.1;                                             //SET INITIAL BIAS TO 0.1
        }
    }
}

void Network::forwardPropagate(double* input_vals){
    layers[0].setValues(input_vals);
    for (int i = 1; i < layer_count; i++){
        double* next_layer = layers[i - 1].computeNextLayer(weights[i-1], biases[i-1], layers[i].getCount());
        layers[i].setValues(next_layer);
        delete next_layer;
        layers[i].activate();
    }
}

void Network::showActiveValues(){
    for (int i = 0; i < layer_count; i++){
        std::cout << "Layer " << i << ":" << std::endl;
        layers[i].showActiveValues();
    }
}

Network::~Network(){
    delete[] layers;
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
    delete[] neuron_counts;
}