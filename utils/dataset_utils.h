#ifndef DATASET_UTILS_H
#define DATASET_UTILS_H

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../neural_network.h"

// Dataset type structure
typedef struct
{
    size_t nb_sets;

    size_t nb_inputs;
    size_t nb_outputs;

    double **input;
    double **target;
} dataset;

// Initialize a dataset
dataset init_dataset(size_t nb_sets, size_t nb_inputs, size_t nb_outputs);

// Frees a dataset from the heap allocated by init_dataset
void free_dataset(dataset ds);

// Shuffles a dataset
void shuffle(dataset ds);

// Test network with a random set in a dataset
void test_random_set(dataset ds, network net);

// Test network accuracy with all the sets in a dataset
void test_accuracy(dataset ds, network net);

// Prints an array
void print_array(double *array, size_t size);

// Converts an array into a digit
char get_digit(double *array, size_t size);

// Get the prediction percentage of an output array
double get_confidence(double *array, size_t size);

#endif
