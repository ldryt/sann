#include "semeion_utils.h"

// Build a semeion dataset
dataset build_semeion(char *path)
{
    FILE *file = fopen(path, "r");
    if (file == NULL)
        errx(EXIT_FAILURE, "Could not open %s\n", path);

    size_t nb_sets = ln_file(file);
    dataset ds = init_dataset(nb_sets, 16 * 16, 10);

    for (size_t l = 0; l < nb_sets; l++)
    {
        char *line = get_line(file);
        parse_semeion(ds, line, l);
        free(line);
    }

    fclose(file);
    return ds;
}

// Parses inputs and targets values from a row in the semeion dataset
void parse_semeion(dataset ds, char *line, size_t row)
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

// Returns the number of lines in a file
size_t ln_file(FILE *file)
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
char *get_line(FILE *file)
{
    int c = EOF;
    size_t i = 0;
    size_t size = 128;
    char *line = (char *)malloc(size * sizeof(char));

    while ((c = getc(file)) != '\n' && c != EOF)
    {
        line[i++] = c;
        if (i + 1 == size)
            line = (char *)realloc(line, (size *= 2) * sizeof(char));
    }

    line[i] = '\0';
    return line;
}
