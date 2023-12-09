#include <stdio.h>
#include "mnist_utils.h"

// Check if system is big-endian
int32_t is_bigendian()
{
    int32_t i = 1;
    char *p = (char *)&i;

    if (p[0] == 1)
        return 0;
    else
        return 1;
}

// Swap endianness of an 32 bit integer
uint32_t swap_endianness(uint32_t n)
{
    unsigned char c[4];

    if (is_bigendian())
        return n;

    else
    {
        c[0] = n & 255;
        c[1] = (n >> 8) & 255;
        c[2] = (n >> 16) & 255;
        c[3] = (n >> 24) & 255;

        return ((uint32_t)c[0] << 24) + ((uint32_t)c[1] << 16)
               + ((uint32_t)c[2] << 8) + c[3];
    }
}

// Build a mnist dataset
dataset build_mnist(char *images_path, char *labels_path)
{
    FILE *fp_images;
    FILE *fp_labels = fopen(labels_path, "r");
    uint32_t img_buffer[4];
    uint32_t lbl_buffer[2];
    size_t nb_sets;
    size_t nb_inputs;

    fp_images = fopen(images_path, "r");
    if (!fp_images)
        errx(EXIT_FAILURE,
            "parse_mnist: error while opening images dataset %s", images_path);

    fp_labels = fopen(labels_path, "r");
    if (!fp_labels)
        errx(EXIT_FAILURE,
            "parse_mnist: error while opening labels dataset %s", labels_path);

    // I could use a buffer of 4 bytes and directly swap each byte, but I
    // wanted to implement the idea behind this nice article:
    // https://developer.ibm.com/articles/au-endianc/
    fread(img_buffer, sizeof(uint32_t), 4, fp_images);
    if (swap_endianness(img_buffer[0]) != 2051)
        errx(EXIT_FAILURE,
            "parse_mnist: bad magic number for images dataset (got 0x%08x)",
            swap_endianness(img_buffer[0]));

    fread(lbl_buffer, sizeof(uint32_t), 2, fp_labels);
    if (swap_endianness(lbl_buffer[0]) != 2049)
        errx(EXIT_FAILURE,
            "parse_mnist: bad magic number for labels dataset (got 0x%08x)",
            swap_endianness(lbl_buffer[0]));

    if (img_buffer[1] != lbl_buffer[1])
        errx(EXIT_FAILURE,
            "parse_mnist: inconsistencies between labels and images datasets");

    nb_sets = swap_endianness(lbl_buffer[1]);
    nb_inputs
        = swap_endianness(img_buffer[2]) * swap_endianness(img_buffer[3]);
    dataset ds = init_dataset(nb_sets, nb_inputs, 10);

    for (size_t set = 0; set < nb_sets; set++)
        parse_mnist_data(ds, images_path, labels_path, set);

    fclose(fp_images);
    fclose(fp_labels);

    return ds;
}

// Parses inputs and targets values from the image and label number 'set' in
// the mnist dataset
void parse_mnist_data(
    dataset ds, char *images_path, char *labels_path, size_t set)
{
    FILE *fp_images = fopen(images_path, "r");
    FILE *fp_labels = fopen(labels_path, "r");
    unsigned char c;

    fseek(fp_images, 4 * sizeof(uint32_t) + set * ds.nb_inputs, SEEK_SET);
    for (size_t i = 0; i < ds.nb_inputs; i++)
    {
        c = fgetc(fp_images);
        ds.input[set][i] = (double)c;
    }

    fseek(fp_labels, 2 * sizeof(uint32_t) + set, SEEK_SET);
    c = fgetc(fp_labels);
    for (size_t k = 0; k < ds.nb_outputs; k++)
        ds.target[set][k] = (double)(c == k);

    fclose(fp_images);
    fclose(fp_labels);
}

/*
int main()
{
    char *si = "./datasets/mnist/train-images.idx3-ubyte";
    char *sl = "./datasets/mnist/train-labels.idx1-ubyte";
    dataset ds = build_mnist(si, sl);
    int set = 5999;
    for (size_t i = 0; i < ds.nb_outputs; i++)
        printf("%lf\n", ds.target[set][i]);
    for (size_t i = 0; i < ds.nb_inputs; i++)
        printf("%03.0lf%s", ds.input[set][i],
            (i + 1) % 28 == 0 ? "\n" : " ");
    return 0;
}
*/