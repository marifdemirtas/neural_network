/*
Student Name: Mehmet Arif Demirtaş
Student ID : 150180001

COMPILE COMMAND: g++ 150180001.cpp

#include "NN.cpp" can be commented out to link two cpp files in compile phase, using 'g++ -o result NN.cpp 150180001.cpp', which will create a executable called 'result'. 
*/

#include <iostream>
#include <fstream>
#include <string>
#include <armadillo>
#include <chrono>

#include "NN.h"

using namespace std;

int main(int argc, char const *argv[]){

    ifstream inputs(argv[1]); //Open a input file stream using the given argument

    int layer_count;
    inputs >> layer_count;

    int* neuron_counts = new int[layer_count];
    for (int i = 0; i < layer_count; ++i)    {
        inputs >> neuron_counts[i];
    }

    int neuron_types[layer_count];
    for (int i = 0; i < layer_count; ++i){
        inputs >> neuron_types[i];
    }

    int test_cases;
    inputs >> test_cases;

    arma::mat x_values(neuron_counts[0], test_cases, arma::fill::zeros);
    if (!x_values.load(argv[2])){
        throw std::logic_error("Invalid values");
    };
    arma::inplace_trans(x_values);

    inputs.close();


    Network* myNN;          //Creates a pointer that will be assigned an object

    try{                    //Creates a neural network if input is valid 
        myNN = new Network(layer_count, neuron_counts, neuron_types);
    }
    catch(string& err){
        cerr << err << endl;
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();
    myNN->forwardPropagate(x_values);   //Does a forward propagation using initial x_values
    auto stop = chrono::high_resolution_clock::now(); 
    
    //myNN->showActiveValues();           //Shows the activated values of nodes after propagation
    
    if (argc > 3){
        myNN->saveOutput(string(argv[3]));
        myNN->saveWeights(string(argv[3]));
        myNN->saveBiases(string(argv[3]));
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start); 
    cout << duration.count() << endl; 


    delete myNN;

    return 0;
}
