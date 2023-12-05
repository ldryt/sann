#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <stdlib.h>

// The network structure.
typedef struct
{
    size_t nb_inputs;
    size_t nb_outputs;
    size_t nb_weights;
    size_t nb_hidden_neurons;

    double *biases;
    double *weights;

    // Hidden layer weights and output layer weights.
    double *ho_weights;

    double *hidden_layer;
    double *output_layer;

    // Always 2 : we didn't implement more than 1 hidden layer yet.
    size_t nb_biases;
} network;

// Create and initialize a network structure.
// Allocates memory on the heap.
network init_network(
    size_t nb_inputs, size_t nb_hidden_neurons, size_t nb_outputs);

// Frees memory on the heap allocated by init_network.
void free_network(network net);

// Back propagation : mesures error changes and adjusts each weights.
void back_propagation(
    network net, double *input, double *target_output, double learning_rate);

// Forward propagation : calculates neuron values with a given input.
void forward_propagation(network net, double *in);

// Returns output of the network with a given input.
double *feed(network net, double *input);

// Total error rate, compares target to actual output with the error function.
double error_rate(
    double *target_output, double *output_layer, size_t nb_outputs);

// Performs forward and back propagation.
// Returns the error rate.
double train(
    network net, double *input, double *target_output, double learning_rate);

#endif
