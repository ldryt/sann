#include "dataset_utils.h"

// Initialize a dataset
dataset init_dataset(size_t nb_sets, size_t nb_inputs, size_t nb_outputs)
{
    dataset ds;

    ds.nb_sets = nb_sets;
    ds.nb_inputs = nb_inputs;
    ds.nb_outputs = nb_outputs;

    double **input = (double **)malloc(nb_sets * sizeof(double *));
    double **target = (double **)malloc(nb_sets * sizeof(double *));
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < nb_sets; j++)
        {
            input[j] = (double *)malloc(nb_inputs * sizeof(double));
            target[j] = (double *)malloc(nb_outputs * sizeof(double));
        }
    }
    ds.input = input;
    ds.target = target;

    return ds;
}

// Frees a dataset from the heap allocated by init_dataset
void free_dataset(dataset ds)
{
    for (size_t set = 0; set < ds.nb_sets; set++)
    {
        free(ds.input[set]);
        free(ds.target[set]);
    }
    free(ds.input);
    free(ds.target);
}

// Shuffles a dataset
void shuffle(dataset ds)
{
    for (size_t set = 0; set < ds.nb_sets; set++)
    {
        size_t random_set = rand() % ds.nb_sets;

        double *input_tmp = ds.input[set];
        ds.input[set] = ds.input[random_set];
        ds.input[random_set] = input_tmp;

        double *target_tmp = ds.target[set];
        ds.target[set] = ds.target[random_set];
        ds.target[random_set] = target_tmp;
    }
}

// Prints an array
void print_array(double *array, int size)
{
    for (int i = 0; i < size; i++)
        printf("%f ", (double)array[i]);
    printf("\n");
}
