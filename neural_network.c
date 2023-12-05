#include "neural_network.h"
#include "math_utils.h"

// Create and initialize a network structure.
// Allocates memory on the heap.
network init_network(
    size_t nb_inputs, size_t nb_hidden_neurons, size_t nb_outputs)
{
    network net = {.nb_inputs = nb_inputs,
        .nb_outputs = nb_outputs,
        .nb_weights = nb_hidden_neurons * (nb_inputs + nb_outputs),
        .nb_hidden_neurons = nb_hidden_neurons,
        .nb_biases = 2};

    net.biases = (double *)malloc(net.nb_biases * sizeof(*net.biases));
    net.weights = (double *)malloc(net.nb_weights * sizeof(*net.weights));
    net.ho_weights = net.weights + nb_hidden_neurons * nb_inputs;
    net.hidden_layer
        = (double *)malloc(nb_hidden_neurons * sizeof(*net.hidden_layer));
    net.output_layer
        = (double *)malloc(nb_outputs * sizeof(*net.output_layer));

    for (size_t i = 0; i < net.nb_weights; i++)
        net.weights[i] = random_d();
    for (size_t i = 0; i < net.nb_biases; i++)
        net.biases[i] = random_d();

    return net;
}

// Frees memory on the heap allocated by init_network.
void free_network(network net)
{
    free(net.weights);
    free(net.biases);
    free(net.hidden_layer);
    free(net.output_layer);
}

// Forward propagation : calculates neuron values with a given input.
void forward_propagation(network net, double *input)
{
    // Calculates hidden layer neuron values
    for (size_t i = 0; i < net.nb_hidden_neurons; i++)
    {
        double res = 0.0;
        for (size_t j = 0; j < net.nb_inputs; j++)
        {
            res += input[j] * net.weights[i * net.nb_inputs + j];
        }
        net.hidden_layer[i] = sigmoid_activation(res + net.biases[0]);
    }

    // Calculates output layer neuron values
    for (size_t i = 0; i < net.nb_outputs; i++)
    {
        double res = 0.0;
        for (size_t j = 0; j < net.nb_hidden_neurons; j++)
        {
            res += net.hidden_layer[j]
                   * net.ho_weights[i * net.nb_hidden_neurons + j];
        }
        net.output_layer[i] = sigmoid_activation(res + net.biases[1]);
    }
}

// Back propagation : calculates error changes and adjusts each weights
void back_propagation(
    network net, double *input, double *target_output, double learning_rate)
{
    for (size_t i = 0; i < net.nb_hidden_neurons; i++)
    {
        // Mesure error changes
        double res = 0;
        for (size_t j = 0; j < net.nb_outputs; j++)
        {
            double x = loss_prime(net.output_layer[j], target_output[j]);
            double y = sigmoid_prime(net.output_layer[j]);

            res += x * y * net.ho_weights[j * net.nb_hidden_neurons + i];

            // Adjust hidden--output weights
            net.ho_weights[j * net.nb_hidden_neurons + i]
                -= x * y * net.hidden_layer[i] * learning_rate;
        }

        // Adjust input--hidden weights
        for (size_t j = 0; j < net.nb_inputs; j++)
        {
            net.weights[i * net.nb_inputs + j]
                -= res * sigmoid_prime(net.hidden_layer[i]) * input[j]
                   * learning_rate;
        }
    }
}

// Returns output of the network with a given input.
double *feed(network net, double *input)
{
    forward_propagation(net, input);
    return net.output_layer;
}

// Total error rate, compares target to actual output with the error function.
double error_rate(
    double *target_output, double *output_layer, size_t nb_outputs)
{
    double res = 0.0;
    for (size_t i = 0; i < nb_outputs; i++)
        res += loss_func(target_output[i], output_layer[i]);
    return res;
}

// Performs forward and back propagation.
// Returns the error rate.
double train(
    network net, double *input, double *target_output, double learning_rate)
{
    forward_propagation(net, input);
    back_propagation(net, input, target_output, learning_rate);
    return error_rate(target_output, net.output_layer, net.nb_outputs);
}
