/*
Student Name: Mehmet Arif Demirtaş
Student ID : 150180001
*/

#include <iostream>
#include <armadillo>
#include <string>

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

void Layer::setValues(arma::mat z_vals)
{
    this->z_vals = z_vals;
    this->a_vals = this->z_vals;
}

void Layer::showActiveValues()
{
    std::cout << a_vals << std::endl;    
}

void Layer::computeZVals(arma::mat weight, arma::vec bias, arma::mat a_vals)
{
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
        weights(i).fill(0.1); // comment out 
        biases(i) = arma::randn(neuron_counts[i]);
        biases(i).fill(0.1);  // comment out
    }


}

arma::mat Network::forwardPropagate(double* input_vals){
    layers[0]->setValues(input_vals);
    for (int i = 1; i < layer_count; i++){
        layers[i]->computeZVals(weights(i), biases(i), layers[i-1]->getA());
        layers[i]->activate();
    }
    return layers[layer_count-1]->getA();
}

arma::mat Network::forwardPropagate(arma::mat input_vals){
    assert(input_vals.n_rows == layers[0]->getCount());

    layers[0]->setValues(input_vals);
    for (int i = 1; i < layer_count; i++){
        layers[i]->computeZVals(weights(i), biases(i), layers[i-1]->getA());
        layers[i]->activate();
    }
//    arma::vec y(layers[layer_count-1]->getA().size(), arma::fill::ones);
//    std::cout << computeLogCost(y) << std::endl;
    return layers[layer_count-1]->getA();
}

double Network::computeLogCost(arma::vec y_val)
{
    return arma::as_scalar(-(1/layers[layer_count-1]->getA().size()) * ((y_val.t() * arma::log(layers[layer_count-1]->getA().t())) + ((1 - y_val.t()) * arma::log(1 - layers[layer_count-1]->getA().t())))); 
}

void Network::showActiveValues(){
    for (int i = 0; i < layer_count; i++){
        std::cout << "Layer " << i << ":" << std::endl;
        layers[i]->showActiveValues();
    }
}

void Network::saveOutput(std::string filename)
{
    layers[layer_count - 1]->getA().save(filename + "/output.txt", arma::arma_ascii);
}

void Network::saveWeights(std::string filename)
{
    weights.save(filename + "/weights.txt"); //For machine to load
    for (int i = 0; i < layer_count; ++i){
        weights(i).save(filename + "/weight_" + std::to_string(i) + ".txt", arma::arma_ascii);
    }
}

void Network::saveBiases(std::string filename)
{
    biases.save(filename + "/biases.txt");
    for (int i = 0; i < layer_count; ++i){
        biases(i).save(filename + std::string("/bias_") + std::to_string(i) + ".txt", arma::arma_ascii);
    }
}


Network::~Network(){
    for (int i = 0; i < layer_count; ++i){
        delete layers[i];
    }
    delete[] layers;
    delete[] neuron_counts;
}