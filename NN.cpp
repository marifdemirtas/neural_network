#include <iostream>
#include <armadillo>
#include <cassert>

#include "NN.h"

/**
 * Helper function for data with multiple classes, turns a vector with tags into an expanded binary vector.
 *
 * @param raw {arma::mat} - row vector to be extended 
 */
arma::mat extendVector(arma::mat raw, int row_size)
{
    arma::mat extended(row_size, raw.n_cols, arma::fill::zeros);
/*
    std::cout << "Gonna replace " << arma::find(raw == 0) << std::endl;
    extended.each_col(arma::find(raw == 0)) = arma::vec({1,0,0});
    std::cout << "extended is "<< std::endl << extended;
    std::cout << "Gonna replace " << arma::find(raw == 1) << std::endl;
    extended.each_col(arma::find(raw == 1)) = arma::vec({0,1,0});
    std::cout << "extended is " << std::endl<< extended;
    std::cout << "Gonna replace " << arma::find(raw == 2) << std::endl;
    extended.each_col(arma::find(raw == 2)) = arma::vec({0,0,1});
    std::cout << "extended is " << std::endl<< extended;
*/

    arma::vec zero_vector(row_size, arma::fill::zeros);
    for (int i = 0; i < row_size; ++i){
        zero_vector(i) = 1;
        extended.each_col(arma::find(raw == i)) = zero_vector;
        zero_vector(i) = 0;
    }

    return extended;
}

/** 
 * Helper function for SigmoidLayer that calculates a vectorized sigmoid function.
 *
 * @param x {arma::mat&} - A reference to an armadillo matrix.
 */
arma::mat SigmoidLayer::sigmoid(arma::mat& x)
{
    return 1 / (1 + arma::exp(-x));
}

void SigmoidLayer::activate()
{
    a_vals = sigmoid(z_vals);
};

void ReluLayer::activate()
{
    a_vals = arma::max(arma::zeros(z_vals.n_rows, z_vals.n_cols), z_vals);
};

void LReluLayer::activate()
{
    a_vals = arma::max(z_vals * RELU_LEAK, z_vals);
};

arma::mat SigmoidLayer::derivate()
{   //dsigmoid is sigmoid * (1 - sigmoid)
    return sigmoid(z_vals) % (1 - sigmoid(z_vals));
};

arma::mat ReluLayer::derivate()
{   // 1 if z_vals > 0, else 0
    arma::mat ret = z_vals;
    ret.elem(find(ret > 0)).fill(1);
    ret.elem(find(ret <= 0)).fill(0);
    return ret;
};

arma::mat LReluLayer::derivate()
{   //1 if z_vals > 0, else RELU_LEAK
    arma::mat ret = z_vals;
    ret.elem(find(ret > 0)).fill(1);
    ret.elem(find(ret <= 0)).fill(RELU_LEAK);
    return ret;
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

void Layer::setValues(arma::mat& z_vals)
{
    this->z_vals = z_vals;
    this->a_vals = this->z_vals;
}

void Layer::showActiveValues()
{
    std::cout << "z: " << z_vals << std::endl;    
    std::cout << "a: " << a_vals << std::endl;    
}

/** 
 * Computes the Z-val for a layer given assoc. weight, bias and previous activations.
 *
 * @param weight {arma::mat} - Weight matrix of the layer where each row represents a neuron in the layer.
 * @param bias {arma::vec} - Bias vector of the layer.
 * @param a_vals {arma::mat} - A matrix where the activation values of the previous layer is represented as columns.
 */
void Layer::computeZVals(arma::mat& weight, arma::vec& bias, arma::mat a_vals)
{
    assert(weight.n_rows == z_vals.n_rows);
    assert(weight.n_cols == a_vals.n_rows);
    assert(bias.n_rows == weight.n_rows);

    this->z_vals = weight * a_vals;
    this->z_vals.each_col() += bias;
}

Network::Network(int layer_count, std::vector<int> neuron_counts, std::vector<int> neuron_types):weights(layer_count), biases(layer_count)
{

    this->layer_count = layer_count;

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
        weights(i) = arma::randn(neuron_counts[i], neuron_counts[i-1]) * 0.01;
        grad_weights = weights;
        biases(i) = arma::randn(neuron_counts[i]) * 0.01;
        grad_biases = biases;
    }

}

/** 
 * Performs a forward propagation using the given input and returns the result.
 *
 * @param x {double*} - A pointer to an array that holds feature values of an input.
 * @returns output {arma::mat} - A vector that holds the related output.
 */
arma::mat Network::forwardPropagate(double* input_vals){
    layers[0]->setValues(input_vals);

    for (int i = 1; i < layer_count; i++){
        layers[i]->computeZVals(weights(i), biases(i), layers[i-1]->getA());
        layers[i]->activate();
    }
    return getOutput();
}

/** 
 * Performs a forward propagation using the given input and returns the result.
 *
 * @param x {arma::mat} - A matrix where the features of each training example is represented as columns.
 * @returns output {arma::mat} - A matrix where the output of each training example is represented as columns.
 */
arma::mat Network::forwardPropagate(arma::mat input_vals)
{
    assert(input_vals.n_rows == layers[0]->getCount());

    layers[0]->setValues(input_vals);
    for (int i = 1; i < layer_count; i++){
        layers[i]->computeZVals(weights(i), biases(i), layers[i-1]->getA());
        layers[i]->activate();
    }
    return getOutput();
}

void Network::backPropagate(arma::mat output_vals)
{
    assert(output_vals.n_rows == getOutput().n_rows);
    assert(output_vals.n_cols == getOutput().n_cols);

/*** 
 *  For start, dJ/dA_i * dA_i/dZ_i * dZ_i/dW_i
 *   1/m           d_cost*derivate()  A_i-1.t()
 *                       of layer
 *  For iteration, also calculate dJ/dZ_i * dZ_i/dA_i-1
 *              W.t()  * (d_cost*derivate())
 *  In iteration, calculate dJ/dW_i-1 as dJ/dA_i-1 * dA_i-1/d_Zi-1 * dZ_i-1/dW_i-1
 *                                        above       derivate()       A.t()
 * 
 *  Similarly, start dJ/dA_i * dA_i/dZ_i * dZ_i/db_i
 *   1/m              d_cost*derivate() -> sum all rows to a column vec
 *  Iteration is above * derivate() -> sum to colvec
 */

//    arma::mat d_cost = -(output_vals / getOutput() + (1 - output_vals)/(1 - getOutput()));
    arma::mat d_activation = getOutput() - output_vals;
    double size_factor = 1.0 / output_vals.n_cols;
    for (int i = layer_count-1; i > 0; i--){
        grad_weights(i) = size_factor * (d_activation * layers[i-1]->getA().t());
        grad_biases(i) = size_factor * arma::sum(d_activation, 1);
        d_activation = weights(i).t() * d_activation;
        d_activation = d_activation % layers[i-1]->derivate();
    }
}

void Network::optimizeParameters(double learning_rate)
{
    for (int i = layer_count-1; i > 0; i--){
        weights(i) -= learning_rate * grad_weights(i);
        biases(i) -= learning_rate * grad_biases(i);
    }
}

/** 
 * Compute logistic cost over the results of last forward propagation and given output vals.
 *
 * @param y_val {arma::mat} - A matrix where the output of each training example is represented as columns (for single output, a rowvec).
 */
arma::mat Network::computeLogCost(arma::mat y_val)
{
    arma::mat cost = -(1.0/getOutput().n_cols) * ((y_val * arma::log(getOutput().t())) + ((1 - y_val) * arma::log(1 - getOutput().t())));
    return cost.diag();
}

void Network::showActiveValues(){
    for (int i = 0; i < layer_count; i++){
        std::cout << "Layer " << i << ":" << std::endl;
        layers[i]->showActiveValues();
    }
}

arma::mat Network::getOutput()
{
    return layers[layer_count-1]->getA();
}

arma::field<arma::mat>& Network::getWeights()
{
    return weights;
}

arma::field<arma::vec>& Network::getBiases()
{
    return biases;
}

void Network::saveOutput(std::string filename)
{
    getOutput().save(filename + "/output.txt", arma::arma_ascii);
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
}


NetworkModel::NetworkModel(int layer_count, std::vector<int> neuron_counts, std::vector<int> neuron_types, double prediction_limit)
{
    net = new Network(layer_count, neuron_counts, neuron_types);
    this->prediction_limit = prediction_limit;
}

NetworkModel::NetworkModel(std::string filename)
{
    std::ifstream options(filename); //Open a input file stream using the given argument
    int layer_count;
    
    std::vector<int> neuron_counts;
    std::vector<int> neuron_types;
    
    options >> layer_count; 
    options >> std::ws;

    int num;
    std::string line;
    std::getline(options, line);

    std::istringstream stream_1(line);
    while(stream_1 >> num){
        neuron_counts.push_back(num);
    }

    std::getline(options, line);
    //stream.str(line); //For some reason, .str(string) does not work
    std::istringstream stream_2(line);
    while(stream_2 >> num){
        neuron_types.push_back(num);
    }

    options.close();

    net = new Network(layer_count, neuron_counts, neuron_types);
}

arma::mat NetworkModel::train(arma::mat input_vals, arma::mat output_vals, int epochs, double learning_rate, bool verbose)
{
    for (int i = 0; i < epochs; ++i){
        net->forwardPropagate(input_vals);
        if (verbose && i % 100 == 0){
            std::cout << "The cost at epoch " << i << " is: " << net->computeLogCost(output_vals) << std::endl;
        }
        net->backPropagate(output_vals);
        net->optimizeParameters(learning_rate);
    }
    arma::mat final_cost = net->computeLogCost(output_vals);
    if (verbose){
        std::cout << "Network trained, final cost: " << final_cost << std::endl;
    }
    return final_cost;
}

std::tuple<arma::mat, double> NetworkModel::test(arma::mat input_vals, arma::mat output_vals)
{
    net->forwardPropagate(input_vals);
    if (output_vals.n_rows == 1){
        arma::uvec errors = arma::find(predictSingleFeature(input_vals) != output_vals);
        arma::mat test_cost = net->computeLogCost(output_vals);
        double accuracy = 1 - (errors.size() / static_cast<double>(output_vals.size()));
        return std::tuple<arma::mat, double>(test_cost, accuracy);
    }else{
        arma::uvec errors = arma::find(predictMultipleFeature(input_vals) != output_vals);
        arma::mat test_cost = net->computeLogCost(output_vals);
        double accuracy = 1 - (errors.size() / static_cast<double>(output_vals.size()));
        return std::tuple<arma::mat, double>(test_cost, accuracy);        
    }
}

arma::mat NetworkModel::predictSingleFeature(arma::mat input_vals)
{
    arma::mat output = net->forwardPropagate(input_vals);
    output.elem(arma::find(output > prediction_limit)).fill(1);
    output.elem(arma::find(output <= prediction_limit)).fill(0);
    return output;
}

arma::umat NetworkModel::predictMultipleFeature(arma::mat input_vals)
{
    arma::mat output = net->forwardPropagate(input_vals);
    std::cout << output << std::endl;
    arma::umat pred = arma::index_max(output);

    arma::rowvec test(input_vals.n_cols);
    return pred;
}

void NetworkModel::save(std::string directory)
{
    net->saveOutput(directory);
    net->saveWeights(directory);
    net->saveBiases(directory);
}

NetworkModel::~NetworkModel()
{
    delete net;
}
