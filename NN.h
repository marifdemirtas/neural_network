/*
Student Name: Mehmet Arif Demirtaş
Student ID : 150180001
*/

#include <armadillo>

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
        void computeZVals(arma::mat weight, arma::vec bias, arma::mat a_vals);

        virtual void activate() = 0;
};

class SigmoidLayer : public Layer
{
    public:
        void activate();
};

class ReluLayer : public Layer
{
    public:
        void activate();
};

class LReluLayer : public Layer
{
    public:
        void activate();
};

class Network
{
    Layer* layers;
    int layer_count;
    int* neuron_counts;
    arma::field<arma::mat> weights;
    arma::field<arma::vec> biases;

public:
    Network(int layer_count, int* neuron_counts, int* neuron_types);
    ~Network();

    void forwardPropagate(double* input_vals);
    void showActiveValues();
};

#endif