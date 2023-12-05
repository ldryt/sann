#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#include <math.h>
#include <stdlib.h>

// Returns a random double between 0 and 1. (inclusive)
double random_d();

// Loss function (mean squared error loss).
double loss_func(double x, double y);

// Derivative of the loss function.
double loss_prime(double x, double y);

// Sigmoid function.
double sigmoid_activation(double z);

// Derivative of the sigmoid function.
double sigmoid_prime(double z);

#endif
