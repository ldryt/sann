#include <time.h>
#include <stdio.h>
#include <err.h>
#include <string.h>
#include <stdlib.h>
#include "neural_network.h"

#define EPOCHS          300
#define NB_INPUTS       256 // 16x16
#define NB_HIDDEN       32
#define NB_OUTPUTS      10
#define LEARNING_RATE   1.7
#define LRATE_MODIFIER  0.999

// Dataset structure
typedef struct
{
    double** input;
    double** target;
    // Number of rows in file (number of sets in dataset)
    int nb_rows;
}
dataset;

// Returns the number of lines in a file
int nb_rows(FILE *file)
{
    int c = EOF;
    int prev = '\n';
    int nb_lines = 0;
    while ((c = getc(file)) != EOF)
    {
        if (c == '\n')
            nb_lines++;
        prev = c;
    }
    if (prev != '\n')
        nb_lines++;

    rewind(file);
    return nb_lines;
}

// Returns a line from a file
char* get_line(FILE *file)
{
    int c = EOF;
    int i = 0;
    int size = 128;
    char *line = (char*) malloc(size * sizeof(char));

    while((c = getc(file)) != '\n' && c != EOF)
    {
        line[i++] = c;
        if (i + 1 == size)
            line = (char*) realloc(line, (size *= 2) * sizeof(char));
    }

    line[i] = '\0';
    return line;
}

// Initialize a dataset
dataset init_dataset(int rows)
{
    double **input = (double**) malloc(rows * sizeof(double*));
    double **target = (double**) malloc(rows * sizeof(double*));
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            input[j] = (double*) malloc(NB_INPUTS * sizeof(double));
            target[j] = (double*) malloc(NB_OUTPUTS * sizeof(double));
        }
    }

    dataset ds = {
        input,
        target,
        rows
    };

    return ds;
}

// Parses inputs and targets values from a row in the dataset
void parse(dataset ds, char* line, int row)
{
    int cols = NB_INPUTS + NB_OUTPUTS;
    for (int col = 0; col < cols; col++)
    {
        double value = atof(strtok(col == 0 ? line : NULL, " "));
        if (col < NB_INPUTS)
            ds.input[row][col] = value;
        else
            ds.target[row][col - NB_INPUTS] = value;
    }
}

// Frees a dataset from the heap allocated by init_dataset
void free_dataset(dataset ds)
{
    for (int row = 0; row < ds.nb_rows; row++)
    {
        free(ds.input[row]);
        free(ds.target[row]);
    }
    free(ds.input);
    free(ds.target);
}

// Shuffles a dataset
void shuffle(dataset ds)
{
    for (int row = 0; row < ds.nb_rows; row++)
    {
        int random_row = rand() % ds.nb_rows;

        double* input_tmp = ds.input[row];
        ds.input[row] = ds.input[random_row];
        ds.input[random_row] = input_tmp;

        double* target_tmp = ds.target[row];
        ds.target[row] = ds.target[random_row];
        ds.target[random_row] = target_tmp;
    }
}

// Build a dataset
dataset build(char* path)
{
    FILE* file = fopen(path, "r");
    if (file == NULL)
        errx(EXIT_FAILURE, "Could not open %s\n", path);

    int rows = nb_rows(file);
    dataset ds = init_dataset(rows);

    for (int row = 0; row < rows; row++)
    {
        char* line = get_line(file);
        parse(ds, line, row);
        free(line);
    }

    fclose(file);
    return ds;
}

// Prints an array
void print_array(double* array, int size)
{
    for (int i = 0; i < size; i++)
        printf("%f ", (double) array[i]);
    printf("\n");
}

int main()
{
    // srand(time(0));
    dataset ds = build("semeion.data");
    network net = init_network(NB_INPUTS, NB_HIDDEN, NB_OUTPUTS);
    double lrate = LEARNING_RATE;
    for (int i = 0; i < EPOCHS; i++)
    {
        shuffle(ds);

        double error = 0;
        for (int row = 0; row < ds.nb_rows; row++)
        {
            double* input = ds.input[row];
            double* target = ds.target[row];
            error += train(net, input, target, lrate);
        }
        printf("epoch %4d: error rate = %.12f | learning rate = %f\n",
            i, (double) error / ds.nb_rows, (double) lrate);

        lrate *= LRATE_MODIFIER;
    }

    double* input = ds.input[0];
    double* target = ds.target[0];
    double* prediction = feed(net, input);

    printf("\n----\n\n");
    printf("Target:\n");
    print_array(target, NB_OUTPUTS);
    printf("Prediction:\n");
    print_array(prediction, NB_OUTPUTS);

    free_dataset(ds);
    free_network(net);

    return EXIT_SUCCESS;
}
