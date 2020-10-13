#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>

#include <armadillo>
#include "../NN.h"


TEST_CASE("Layer class unit tests")
{
    SigmoidLayer next(3); //create a layer with 5 neurons

    SECTION("Getting before setting"){
        next.getA();
        next.getZ();
    }

    arma::vec x_vals = {1, 2, 3};

    SECTION("Setting values from array"){
        double x_vals_arr[3] = {1, 2, 3};
        next.setValues(x_vals_arr);

        REQUIRE(arma::approx_equal(next.getZ(), x_vals, "absdiff", 0.001));
        REQUIRE(arma::approx_equal(next.getA(), x_vals, "absdiff", 0.001));
    }

    SECTION("Setting values from arma::vec"){
        REQUIRE(arma::approx_equal(next.getZ(), arma::vec(3, arma::fill::ones), "absdiff", 0.001));
        next.setValues(x_vals);
        REQUIRE(arma::approx_equal(next.getZ(), x_vals, "absdiff", 0.001));
        REQUIRE(arma::approx_equal(next.getA(), x_vals, "absdiff", 0.001));
    }

    SECTION("Activating values"){
        arma::vec x_vals = {1, 2, 3};
        next.setValues(x_vals);
        next.activate();
        REQUIRE(!arma::approx_equal(next.getA(), next.getZ(), "absdiff", 0.001));
        arma::mat sigmoid_val =  1 / (1 + arma::exp(-next.getZ()));
        REQUIRE(arma::approx_equal(next.getA(), sigmoid_val, "absdiff", 0.001));
    }

    SigmoidLayer prev(5); //create a layer with 3 neurons
    double ones[5] = {1,1,1,1,1};
    prev.setValues(ones);
    arma::mat w(3, 5); 
    w.fill(0.1);
    arma::vec b(3);
    b.fill(0.1);
        //create and fill the weight and bias matrices to go from prev to next
    SECTION("Forward-prop (single step)"){
        next.computeZVals(w, b, prev.getA());
        arma::vec z_vals = (w * prev.getA());
        z_vals.each_col() += b; 
        REQUIRE(next.getZ().n_cols == 1);
        REQUIRE(next.getZ().n_rows == next.getCount());
        REQUIRE(arma::approx_equal(next.getZ(), z_vals, "absdiff", 0.001));
    }

}

TEST_CASE("Inherited layer class activation tests")
{
    Layer* l;
    arma::vec v({-5, -1, -0.5, 0, 0.5, 1, 5}); //7 values to test activations
    SECTION("Sigmoid layer activation"){
        l = new SigmoidLayer(7);
        l->setValues(v);
        l->activate();
        arma::vec approx_sigmoid({0.006692, 0.268941, 0.377541, 0.5, 0.622459, 0.731059, 0.993307});
        REQUIRE(arma::approx_equal(approx_sigmoid, l->getA(), "absdiff", 0.001));
        delete l;
    }

    SECTION("Relu layer activation"){
        l = new ReluLayer(7);
        l->setValues(v);
        l->activate();
        arma::vec approx_relu({0, 0, 0, 0, 0.5, 1, 5});
        REQUIRE(arma::approx_equal(approx_relu, l->getA(), "absdiff", 0.001));
        delete l;
    }

    SECTION("Leaky Relu layer activation"){
        l = new LReluLayer(7);
        l->setValues(v);
        l->activate();
        arma::vec approx_lrelu({-0.5, -0.1, -0.05, 0, 0.5, 1, 5});
        REQUIRE(arma::approx_equal(approx_lrelu, l->getA(), "absdiff", 0.001));
        delete l;
    }


}


TEST_CASE("Network class unit tests")
{
    int layer_count = 3;
    int* neuron_counts = new int[3]{4, 2, 1};
    int neuron_types[3] = {2, 2, 2};
    Network NN(layer_count, neuron_counts, neuron_types);

    SECTION("Forward-prop"){
        arma::mat inputs(4, 1, arma::fill::ones);
        arma::mat a_vals = NN.forwardPropagate(inputs);
        REQUIRE(arma::approx_equal(a_vals, arma::vec({0.2}), "absdiff", 0.001)); //0.2 calculated by hand for weights=bias=0.1
    }

}