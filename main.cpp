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

    arma::mat x_values;
    if (!x_values.load(argv[2])){
        throw std::logic_error("Invalid values");
    };
    arma::inplace_trans(x_values);

    arma::mat y_values;
    if (!y_values.load(argv[3])){
        throw std::logic_error("Invalid values");
    };
//    arma::inplace_trans(y_values);

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
    

//    myNN->train(x_values, y_values.t(), 30000, 0.01);

    if (argc > 4){
        myNN->saveOutput(string(argv[4]));
        myNN->saveWeights(string(argv[4]));
        myNN->saveBiases(string(argv[4]));
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start); 
    cout << "Done in " << duration.count() << " microseconds" << endl; 


    delete myNN;

    return 0;
}
