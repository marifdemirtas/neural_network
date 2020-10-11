/*
Student Name: Mehmet Arif Demirtaş
Student ID : 150180001
*/

#ifndef _NN
#define _NN

class Neuron //Abstract base class for neurons
{
private:
    double z; //value
    double a; //activated value
public:
    Neuron(double z);
//    ~Neuron();
    const double getZ() const{
        return z;
    };
    const double getA() const{
        return a;
    };
    void setZ(double z){
        this->z = z;
    };
    void setA(double a){
        this->a = a;
    };

    virtual void activate() = 0;
};

class SigmoidNeuron: public Neuron
{
public:
    SigmoidNeuron(double z = 0):Neuron(z){};
//    ~SigmoidNeuron();
    void activate();
};

class ReluNeuron: public Neuron
{
public:
    ReluNeuron(double z = 0):Neuron(z){};
//    ~ReluNeuron();
    void activate();
};

class LReluNeuron: public Neuron
{
public:
    LReluNeuron(double z = 0):Neuron(z){};
//    ~LReluNeuron();
    void activate();
};

class Layer
{
    Neuron* neurons;
    int neuron_count;

public:
    Layer(){
        neurons = NULL;
        neuron_count = 0;
    }
    ~Layer();

    const int getCount() const{
        return neuron_count;
    };

    void init(int neuron_count, int neuron_type);
    void setValues(double* z_vals);
    void activate();
    void showActiveValues();

    double* computeNextLayer(double** weight, double* bias, int next_layer_size);
};

class Network
{
    Layer* layers;
    int layer_count;
    int* neuron_counts;
    double*** weights;
    double** biases;

public:
    Network(int layer_count, int* neuron_counts, int* neuron_types);
    ~Network();

    void forwardPropagate(double* input_vals);
    void showActiveValues();
};

#endif