#include <string>
#include <sstream>
#include <vector>
#include <tuple>

#include <armadillo>

#ifndef _NN
#define _NN
#define RELU_LEAK 0.1

arma::mat extendVector(arma::mat raw, int row_size);

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
        void setValues(arma::mat& z_vals);
        void computeZVals(arma::mat& weight, arma::vec& bias, arma::mat a_vals);

        virtual void activate() = 0;
        virtual arma::mat derivate() = 0;
};

class SigmoidLayer : public Layer
{
        arma::mat sigmoid(arma::mat& x);
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

    arma::field<arma::mat> weights;
    arma::field<arma::vec> biases;

    arma::field<arma::mat> grad_weights;
    arma::field<arma::vec> grad_biases;

public:
    Network(int layer_count, std::vector<int> neuron_counts, std::vector<int> neuron_types);
    //constructor from file
    ~Network();

    arma::mat forwardPropagate(double* input_vals);
    arma::mat forwardPropagate(arma::mat input_vals); //can return activation values of last layer

    void backPropagate(arma::mat output_vals); //does a round of backprop, sets up grads matrices
    void optimizeParameters(double learning_rate); //automatically updates parameters using the grads

    void setWeights(double val){
        for (int i = 0; i < layer_count; ++i){
            weights(i).fill(0.1);
            biases(i).fill(0.1);
        }
    };

    arma::mat computeLogCost(arma::mat y_val);

    void showActiveValues();

    arma::mat getOutput();
    arma::field<arma::mat>& getWeights();
    arma::field<arma::vec>& getBiases();

    void saveOutput(std::string filename);
    void saveWeights(std::string filename);
    void saveBiases(std::string filename);
};


class NetworkModel
{
    double prediction_limit;
public:
    Network* net;
    NetworkModel(int layer_count, std::vector<int> neuron_counts, std::vector<int> neuron_types, double prediction_limit = 0.5);
    NetworkModel(std::string filename);

    arma::mat train(arma::mat input_vals, arma::mat output_vals, int epochs, double learning_rate, bool verbose=false);
    std::tuple<arma::mat, double> test(arma::mat input_vals, arma::mat output_vals);
    arma::mat predictSingleFeature(arma::mat input_vals);
    arma::umat predictMultipleFeature(arma::mat input_vals);

    void save(std::string directory);

    ~NetworkModel();
    
};

#endif