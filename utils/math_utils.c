#include "math_utils.h"

// Returns a random double between -0.5 and 0.5
double random_d()
{
    return rand() / (double)RAND_MAX - 0.5;
}

// Sigmoid function.
double sigmoid_activation(double z)
{
    return 1.0 / (1.0 + expf(-z));
}

// Derivative of the sigmoid function.
double sigmoid_prime(double z)
{
    return z * (1.0 - z);
}

// Loss function (mean squared error loss).
double loss_func(double x, double y)
{
    return 0.5 * (double)pow(x - y, 2);
}

// Derivative of the loss function.
double loss_prime(double x, double y)
{
    return x - y;
}
