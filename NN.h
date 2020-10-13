/*
Student Name: Mehmet Arif Demirtaş
Student ID : 150180001
*/

#include <armadillo>
#include <string>

#ifndef _NN
#define _NN
#define RELU_LEAK 0.1

class Layer //abstract base class for layers of different activations
{
        int neuron_count;

    protected:
        arma::mat z_vals;
        arma::mat a_vals;

    public:
        Layer(int neuron_count);

        const int getCount() const{
            return neuron_count;
        };

        arma::mat getA(){
            return a_vals;
        }

        arma::mat getZ(){
            return z_vals;
        }

        void showActiveValues();
        void setValues(double* z_vals);
        void setValues(arma::mat z_vals);
        void computeZVals(arma::mat weight, arma::vec bias, arma::mat a_vals);

        virtual void activate() = 0;
        virtual arma::mat derivate() = 0;
};

class SigmoidLayer : public Layer
{
        arma::mat sigmoid(arma::mat x);
    public:
        SigmoidLayer(int count):Layer(count){};
        void activate();
        arma::mat derivate();
};

class ReluLayer : public Layer
{
    public:
        ReluLayer(int count):Layer(count){};
        void activate();
        arma::mat derivate();
};

class LReluLayer : public Layer
{
    public:
        LReluLayer(int count):Layer(count){};
        void activate();
        arma::mat derivate();
};

class Network
{
    Layer** layers;
    int layer_count;
    int* neuron_counts;
    arma::field<arma::mat> weights;
    arma::field<arma::vec> biases;

    arma::field<arma::mat> grad_weights;
    arma::field<arma::vec> grad_biases;

public:
    Network(int layer_count, int* neuron_counts, int* neuron_types);
    //constructor from file
    ~Network();

    arma::mat forwardPropagate(double* input_vals);
    arma::mat forwardPropagate(arma::mat input_vals); //can return activation values of last layer

    void backPropagate(arma::mat output_vals); //does a round of backprop, sets up grads matrices
    void optimizeParameters(double learning_rate); //automatically updates parameters using the grads


    //setWeights(arma::mat)
    //setBias

    double computeLogCost(arma::vec y_val);
    //computeLogCost
    //get a y_values in main

    //WRAPPER CLASS THAT will implement grad desc.
    void train(arma::mat input_vals, arma::mat output_vals, int epochs, double learning_rate);

    void showActiveValues();

    arma::mat getOutput();
    arma::field<arma::mat> getWeights();
    arma::field<arma::vec> getBiases();

    void saveOutput(std::string filename);
    void saveWeights(std::string filename);
    void saveBiases(std::string filename);
};

#endif