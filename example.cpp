#include <armadillo>
#include <iostream>
#include "NN.h"

using namespace std;
using namespace arma;

arma::mat extendVector(arma::mat raw, int row_size);

int main(int argc, char const *argv[])
{
    rowvec x;
    x.load("files/iris/test-y.txt");
    rowvec t({1, 2, 0, 0, 2});
    extendVector(t, 3);
    return 0;
}