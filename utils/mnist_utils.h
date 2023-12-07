#ifndef MNIST_UTILS_H
#define MNIST_UTILS_H

#include <stdint.h>
#include <stdio.h>
#include "dataset_utils.h"

// Check if system is big-endian
int is_bigendian();

// Swap endianness of an 32 bit integer
uint32_t swap_endianness(uint32_t n);

// Build a mnist dataset
dataset build_mnist(char *images_path, char *labels_path);

// Parses inputs and targets values from the image and label number 'set' in
// the mnist dataset
void parse_mnist_data(
    dataset ds, char *images_path, char *labels_path, size_t set);

#endif
