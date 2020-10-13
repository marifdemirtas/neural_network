/*
Student Name: Mehmet Arif Demirtaş
Student ID : 150180001
*/

#include <armadillo>
#include <string>

#ifndef _NN
#define _NN

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
};

class SigmoidLayer : public Layer
{
    public:
        SigmoidLayer(int count):Layer(count){};
        void activate();
};

class ReluLayer : public Layer
{
    public:
        ReluLayer(int count):Layer(count){};
        void activate();
};

class LReluLayer : public Layer
{
    public:
        LReluLayer(int count):Layer(count){};
        void activate();
};

class Network
{
    Layer** layers;
    int layer_count;
    int* neuron_counts;
    arma::field<arma::mat> weights;
    arma::field<arma::vec> biases;

public:
    Network(int layer_count, int* neuron_counts, int* neuron_types);
    //constructor from file
    ~Network();

    arma::mat forwardPropagate(double* input_vals);
    arma::mat forwardPropagate(arma::mat input_vals); //can return activation values of last layer

    //setWeights(arma::mat)
    //setBias

    //modifyWeightAt (span?)
    //modifyBiasAt (span?)

    //backprop (automatically set parameters)

    double computeLogCost(arma::vec y_val);
    //computeLogCost
    //get a y_values in main

    //WRAPPER CLASS THAT will implement grad desc.

    void showActiveValues();

    void saveOutput(std::string filename);
    void saveWeights(std::string filename);
    void saveBiases(std::string filename);
};

#endif