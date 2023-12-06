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
int32_t swap_endianness(int32_t n)
{
    char c[4];

    if (is_bigendian())
        return n;

    else
    {
        c[0] = n & 255;
        c[1] = (n >> 8) & 255;
        c[2] = (n >> 16) & 255;
        c[3] = (n >> 24) & 255;

        return ((int32_t)c[0] << 24) + ((int32_t)c[1] << 16)
               + ((int32_t)c[2] << 8) + c[3];
    }
}

// Parses inputs and targets values from the image number 'set' in
// the mnist dataset
void parse_mnist(dataset ds, char *images_path, char *labels_path, size_t set)
{
    int32_t header_buffer[0];
    unsigned char pixels_buffer[28];

    FILE *fp_images = fopen(images_path, "r");
    FILE *fp_labels = fopen(labels_path, "r");

    // I could use a buffer of 4 bytes and directly swap each byte, but I
    // wanted to implement the idea behind this nice article:
    // https://developer.ibm.com/articles/au-endianc/
    fread(header_buffer, 4, 1, fp_images);
    if (swap_endianness(header_buffer[0]) != 2051)
        errx(EXIT_FAILURE, "parse_mnist: bad magic number (got 0x%08x)",
            swap_endianness(header_buffer[0]));

    // random-looking formula to set position to the image number 'set'
    fseek(fp_images, (4 * 4) + set * (28 * 28), SEEK_SET);

    for (size_t i = 0; i < 28; i++)
    {
        fread(pixels_buffer, 1, 28, fp_images);
        for (size_t j = 0; j < 28; j++)
            ds.input[set][i + j] = pixels_buffer[j];
    }

    fclose(fp_images);
    fclose(fp_labels);
}

int main()
{
    if (is_bigendian())
        printf("%d\n", is_bigendian());
    dataset ds = init_dataset(60000, 28 * 28, 10);
    parse_mnist(ds, "./datasets/mnist/train-images.idx3-ubyte", "./datasets/mnist/train-labels.idx3-ubyte", 0);
    return 0;
}