#include <iostream>
#include <armadillo>
#include <chrono>

#include "NN.h"

using namespace std;

int main(int argc, char const *argv[]){

//    myNN->forwardPropagate(x_values);   //Does a forward propagation using initial x_values

    NetworkModel model(argv[1]);

    arma::mat x;
    arma::mat y;
    x.load(argv[2]);
    y.load(argv[3]);


    if (y.n_rows <= 1){
        y = extendVector(y, 3);
    }

    cout << "Shape of x: " << x.n_rows << "*" << x.n_cols << endl; 
    cout << "Shape of y: " << y.n_rows << "*" << y.n_cols << endl; 

    auto start = chrono::high_resolution_clock::now();
    model.train(x, y, 200, 0.00075, false);
    auto stop = chrono::high_resolution_clock::now(); 
    
    model.save(argv[4]);

    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start); 
    cout << "Training done in " << duration.count() << " microseconds" << endl; 

    arma::mat x_test;
    arma::mat y_test;
    x_test.load(argv[5]);
    y_test.load(argv[6]);

/*
    if (y_test.n_rows <= 1){
        y_test = extendVector(y_test, 3);
    }
*/
    cout << "Shape of x_test: " << x_test.n_rows << "*" << x_test.n_cols << endl; 
    cout << "Shape of y_test: " << y_test.n_rows << "*" << y_test.n_cols << endl; 

    start = chrono::high_resolution_clock::now();
    auto train_results = model.test(x, y);
    auto test_results = model.test(x_test, y_test);
    stop = chrono::high_resolution_clock::now(); 

    cout << "Total cost of train set:" << get<0>(train_results) << endl;
    cout << "Accuracy on train set:" << get<1>(train_results) << endl;

    cout << "Total cost of test set:" << get<0>(test_results) << endl;
    cout << "Accuracy on test set:" << get<1>(test_results) << endl;

    duration = chrono::duration_cast<chrono::microseconds>(stop - start); 
    cout << "Testing done in " << duration.count() << " microseconds" << endl; 


    return 0;
}
