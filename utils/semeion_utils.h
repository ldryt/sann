#ifndef SEMEION_UTILS_H
#define SEMEION_UTILS_H

#include "dataset_utils.h"

// Build a semeion dataset
dataset build_semeion(char *path);

// Parses inputs and targets values from a row in the semeion dataset
void parse_semeion(dataset ds, char *line, size_t row);

// Returns the number of lines in a file
size_t ln_file(FILE *file);

// Returns a line from a file
char *get_line(FILE *file);

#endif
