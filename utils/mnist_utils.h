#ifndef MNIST_UTILS_H
#define MNIST_UTILS_H

#include <stdio.h>
#include <stdint.h>
#include "dataset_utils.h"

// Check if system is big-endian
int is_bigendian();

// Swap endianness of an 32 bit integer
int32_t swap_endianness(int32_t n);

// Parses inputs and targets values from the image number 'set' in
// the mnist dataset
void parse_mnist(dataset ds, char *images_path, char *labels_path, size_t set);

#endif
