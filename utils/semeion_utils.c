#include "semeion_utils.h"

// Initialize a dataset
dataset init_dataset(size_t rows, size_t nb_inputs, size_t nb_outputs)
{
    double **input = (double**) malloc(rows * sizeof(double*));
    double **target = (double**) malloc(rows * sizeof(double*));
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < rows; j++)
        {
            input[j] = (double*) malloc(nb_inputs * sizeof(double));
            target[j] = (double*) malloc(nb_outputs * sizeof(double));
        }
    }

    dataset ds = {
        input,
        target,
        rows,
        nb_inputs,
        nb_outputs
    };

    return ds;
}

// Frees a dataset from the heap allocated by init_dataset
void free_dataset(dataset ds)
{
    for (size_t row = 0; row < ds.nb_rows; row++)
    {
        free(ds.input[row]);
        free(ds.target[row]);
    }
    free(ds.input);
    free(ds.target);
}

// Build a dataset
dataset build(char* path, size_t nb_inputs, size_t nb_outputs)
{
    FILE* file = fopen(path, "r");
    if (file == NULL)
        errx(EXIT_FAILURE, "Could not open %s\n", path);

    size_t rows = nb_rows(file);
    dataset ds = init_dataset(rows, nb_inputs, nb_outputs);

    for (size_t row = 0; row < rows; row++)
    {
        char* line = get_line(file);
        parse(ds, line, row);
        free(line);
    }

    fclose(file);
    return ds;
}

// Shuffles a dataset
void shuffle(dataset ds)
{
    for (size_t row = 0; row < ds.nb_rows; row++)
    {
        size_t random_row = rand() % ds.nb_rows;

        double* input_tmp = ds.input[row];
        ds.input[row] = ds.input[random_row];
        ds.input[random_row] = input_tmp;

        double* target_tmp = ds.target[row];
        ds.target[row] = ds.target[random_row];
        ds.target[random_row] = target_tmp;
    }
}

// Returns the number of lines in a file
size_t nb_rows(FILE *file)
{
    int c = EOF;
    int prev = '\n';
    size_t nb_lines = 0;
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
    size_t i = 0;
    size_t size = 128;
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

// Parses inputs and targets values from a row in the dataset
void parse(dataset ds, char* line, size_t row)
{
    size_t cols = ds.nb_inputs + ds.nb_outputs;
    for (size_t col = 0; col < cols; col++)
    {
        double value = atof(strtok(col == 0 ? line : NULL, " "));
        if (col < ds.nb_inputs)
            ds.input[row][col] = value;
        else
            ds.target[row][col - ds.nb_inputs] = value;
    }
}
