#include <stdio.h>
#include "logum_utils.h"

// Build a logum dataset
dataset build_logum(char *images_folder_path)
{
    dataset ds;
    DIR *dirp;
    struct dirent *entry;
    size_t set;
    size_t nb_sets;

    char img_dir_path[255];
    char lbl_path[255];
    char img_path[255];

    strcpy(img_dir_path, images_folder_path);
    strcpy(lbl_path, images_folder_path);
    strcat(img_dir_path, "/images");
    strcat(lbl_path, "/labels.txt");

    nb_sets = get_nb_files(img_dir_path);
    if (nb_sets != get_nb_lines(lbl_path))
        errx(EXIT_FAILURE,
            "Inconsistencies between number of label sets and image sets "
            "(%ld,%ld)",
            get_nb_lines(lbl_path), nb_sets);

    ds = init_dataset(
        nb_sets, 28 * 28, 11); // TODO: Should not be hard-coded like this

    dirp = opendir(img_dir_path);
    if (!dirp)
        errx(EXIT_FAILURE, "Couldn't open directory %s", img_dir_path);

    while ((entry = readdir(dirp)) != NULL)
    {
        if (strcmp(entry->d_name, "..") == 0
            || strcmp(entry->d_name, ".")
                   == 0) // TODO: find another way (not DT_REG)
            continue;

        memset(img_path, 0, 255);
        strcpy(img_path, img_dir_path);
        strcat(img_path, "/");
        strcat(img_path, entry->d_name);

        set = myatol(entry->d_name);

        parse_logum_data(ds, img_path, lbl_path, set);
    }

    closedir(dirp);

    return ds;
}

// Parses inputs and targets values from the image file and label file number
// 'set' in the logum dataset
void parse_logum_data(
    dataset ds, char *image_path, char *labels_path, size_t set)
{
    FILE *img_f;
    FILE *lbl_f;
    size_t img_width;
    size_t img_height;
    size_t c;

    img_f = fopen(image_path, "r");
    if (!img_f)
        errx(EXIT_FAILURE, "Couldn't open image %s", image_path);

    lbl_f = fopen(labels_path, "r");
    if (!lbl_f)
        errx(EXIT_FAILURE, "Couldn't open image %s", labels_path);

    fscanf(img_f, "P2\n%ld %ld\n255\n", &img_width, &img_height);
    if (img_width != 28
        || img_height != 28) // TODO: Should not be hard-coded like this
        errx(EXIT_FAILURE, "Wrong image size for %s (w=%ld,h=%ld)", image_path,
            img_width, img_height);

    for (size_t i = 0; i < img_width; i++)
    {
        for (size_t j = 0; j < img_height; j++)
        {
            int v;
            fscanf(img_f, "%d ", &v);
            ds.input[set][i * img_height + j] = v;
        }
    }

    for (size_t i = 0; i < set + 1; i++)
        fscanf(lbl_f, "%ld\n", &c);
    for (size_t k = 0; k < ds.nb_outputs; k++)
        ds.target[set][k] = (double)(c == k);

    fclose(img_f);
}

size_t get_nb_files(char *dir_path)
{
    DIR *dirp;
    struct dirent *entry;
    size_t file_count = 0;

    dirp = opendir(dir_path);
    if (!dirp)
        errx(EXIT_FAILURE, "Couldn't open directory %s", dir_path);
    while ((entry = readdir(dirp)) != NULL)
    {
        if (strcmp(entry->d_name, "..") == 0
            || strcmp(entry->d_name, ".")
                   == 0) // TODO: find another way (not DT_REG)
            continue;
        file_count++;
    }

    closedir(dirp);
    return file_count;
}

// Returns the number of lines in a file
size_t get_nb_lines(char *path)
{
    FILE *file;
    int c = EOF;
    int prev = '\n';
    size_t nb_lines = 0;

    file = fopen(path, "r");
    if (!file)
        errx(EXIT_FAILURE, "Couldn't open file %s", path);

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

// Converts string into a unsigned long
size_t myatol(char *str)
{
    size_t num = 0;
    for (int i = 0; str[i] != '\0'; i++)
    {
        if (str[i] < '0' || str[i] > '9')
            continue;
        num = num * 10 + (str[i] - 48);
    }
    return num;
}