/*
Student Name: Mehmet Arif Demirtaş
Student ID : 150180001

COMPILE COMMAND: g++ 150180001.cpp

#include "NN.cpp" can be commented out to link two cpp files in compile phase, using 'g++ -o result NN.cpp 150180001.cpp', which will create a executable called 'result'. 
*/

#include <iostream>
#include <fstream>
#include <armadillo>

#include "NN.h"

using namespace std;

int main(int argc, char const *argv[]){

    ifstream inputs(argv[1]); //Open a input file stream using the given argument

    int layer_count;
    inputs >> layer_count;

    int* neuron_counts = new int[layer_count];
    for (int i = 0; i < layer_count; ++i)
    {
        inputs >> neuron_counts[i];
    }

    int neuron_types[layer_count];
    for (int i = 0; i < layer_count; ++i)
    {
        inputs >> neuron_types[i];
    }

    int test_cases;
    inputs >> test_cases;


    int input_count = 0;
//    double x_values[neuron_counts[0]];
    arma::mat x_values(neuron_counts[0], test_cases, arma::fill::zeros);


    for (int i = 0; i < neuron_counts[0]*test_cases; ++i){
        inputs >> x_values(i);
    }
/*  
    while(inputs >> x_values[input_count++]);
    input_count--;

    try{                    //Checks if there are any input values remaining 
        if(input_count != neuron_counts[0]) 
            throw string("Input shape does not match!");
    }catch(string err){
        cerr << err << endl;
        inputs.close();
        return 1;
    }
*/
    inputs.close();

    Network* myNN;          //Creates a pointer that will be assigned an object

    try{                    //Creates a neural network if input is valid 
        myNN = new Network(layer_count, neuron_counts, neuron_types);
    }
    catch(string& err){
        cerr << err << endl;
        return 1;
    }

    myNN->forwardPropagate(x_values);   //Does a forward propagation using initial x_values
    myNN->showActiveValues();           //Shows the activated values of nodes after propagation

    delete myNN;
    return 0;
}
