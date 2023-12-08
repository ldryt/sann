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

// Test network with a random set in a dataset
void test_random_set(dataset ds, network net)
{
    srand(time(NULL));
    shuffle(ds);

    double *prediction = feed(net, ds.input[0]);
    char prediction_c = get_digit(prediction, ds.nb_outputs);
    char target_c = get_digit(ds.target[0], ds.nb_outputs);

    printf("Input:\n");
    for (size_t i = 0; i < ds.nb_inputs; i++)
        printf("%s%s", ds.input[0][i] > 0 ? "▆" : "-",
            (i + 1) % (size_t)sqrt(ds.nb_inputs) == 0 ? "\n" : " ");
    printf("Target: ");
    printf("expecting %d\n", target_c);
    print_array(ds.target[0], ds.nb_outputs);
    printf("Prediction: ");
    printf("predicted a %d! %s\n", prediction_c,
        target_c == prediction_c ? "Youpi :)" : "Too Bad :(");
    print_array(prediction, ds.nb_outputs);
}

// Test network accuracy with all the sets in a dataset
void test_accuracy(dataset ds, network net)
{
    double accuracy = 0;
    double avg_confidence = 0;
    double *prediction;
    char prediction_c;
    char target_c;
    for (size_t set = 0; set < ds.nb_sets; set++)
    {
        prediction = feed(net, ds.input[set]);
        prediction_c = get_digit(prediction, ds.nb_outputs);
        target_c = get_digit(ds.target[set], ds.nb_outputs);

        if (prediction_c == target_c)
            accuracy++;
        avg_confidence += get_confidence(prediction, ds.nb_outputs);
    }
    accuracy /= ds.nb_sets;
    avg_confidence /= ds.nb_sets;

    printf("Average confidence: %.1lf%%\n", avg_confidence * 100);
    printf("Network accuracy: %.1lf%%\n", accuracy * 100);
}

// Prints an array
void print_array(double *array, size_t size)
{
    for (size_t i = 0; i < size; i++)
        printf("%f ", (double)array[i]);
    printf("\n");
}

// Converts an array into a digit
char get_digit(double *array, size_t size)
{
    size_t res = 0;
    for (size_t i = 1; i < size; i++)
        if (array[i] > array[res])
            res = i;

    return res;
}

// Get the prediction percentage of an output array
double get_confidence(double *array, size_t size)
{
    double res = array[0];
    for (size_t i = 1; i < size; i++)
        if (array[i] > res)
            res = array[i];

    return res;
}