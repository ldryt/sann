#ifndef LOGUM_UTILS_H
#define LOGUM_UTILS_H

#include <dirent.h>
#include <stdio.h>
#include "dataset_utils.h"

// Build a logum dataset
dataset build_logum(char *folder_path);

// Parses inputs and targets values from the image file and label file number
// 'set' in the logum dataset
void parse_logum_data(
    dataset ds, char *image_path, char *labels_path, size_t set);

// Get number of files in a directory
size_t get_nb_files(char *dir_path);

// Returns the number of lines in a file
size_t get_nb_lines(char *path);

// Converts string into a unsigned long
size_t myatol(char *str);

#endif
