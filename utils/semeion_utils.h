#ifndef SEMEION_DATASET_H
#define SEMEION_DATASET_H

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Dataset type structure
typedef struct
{
    double **input;
    double **target;
    // Number of rows in file (number of sets in dataset)
    size_t nb_rows;

    // Neural network related parameters
    size_t nb_inputs;
    size_t nb_outputs;
} dataset;

// Initialize a dataset
dataset init_dataset(size_t rows, size_t nb_inputs, size_t nb_outputs);

// Frees a dataset from the heap allocated by init_dataset
void free_dataset(dataset ds);

// Build a dataset
dataset build(char *path, size_t nb_inputs, size_t nb_outputs);

// Shuffles a dataset
void shuffle(dataset ds);

// Returns the number of lines in a file
size_t nb_rows(FILE *file);

// Returns a line from a file
char *get_line(FILE *file);

// Parses inputs and targets values from a row in the dataset
void parse(dataset ds, char *line, size_t row);

// Prints an array
void print_array(double *array, int size);

#endif
