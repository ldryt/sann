#include "neural_network.h"
#include "utils/mnist_utils.h"
#include "utils/semeion_utils.h"
#include "utils/logum_utils.h"

#define EPOCHS 300
#define HIDDEN_NEURONS 42
#define LEARNING_RATE 0.01
#define LRATE_MODIFIER 0.99
#define BATCH_SIZE 1024

#define DS_MNIST_TRAIN_IMAGES "./datasets/mnist/t10k-images.idx3-ubyte"
#define DS_MNIST_TRAIN_LABELS "./datasets/mnist/t10k-labels.idx1-ubyte"
#define DS_SEMEION_FILE "./datasets/semeion/semeion.data"
#define DS_LOGUM_FOLDER "./datasets/logum/training"

int train_on_xor()
{
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double target[4] = {0, 1, 1, 0};
    network net = init_network(2, 4, 1);
    double error = 0;
    for (size_t i = 0; i < EPOCHS * 1000; i++)
    {
        for (size_t j = 0; j < 4; j++)
            error = train(net, inputs[j], &target[j], LEARNING_RATE);
        if (i % 1000 == 0)
            printf("epoch %05ld: error rate = %.5f (learning rate = %.2f)\n",
                i, error, LEARNING_RATE);
    }

    for (size_t i = 0; i < 4; i++)
    {
        double *prediction = feed(net, inputs[i]);

        printf("--------\n");
        printf("Input:\n");
        printf("{%d,%d}\n", (int)inputs[i][0], (int)inputs[i][1]);
        printf("Target:\n");
        printf("%d\n", (int)target[i]);
        printf("Prediction:\n");
        printf("%f\n", prediction[0]);
    }

    return EXIT_SUCCESS;
}

network train_on_ds(dataset ds)
{
    network net = init_network(ds.nb_inputs, HIDDEN_NEURONS, ds.nb_outputs);

    double lrate = LEARNING_RATE;
    for (size_t i = 0; i < EPOCHS; i++)
    {
        shuffle(ds);

        double error = 0;
        for (size_t set = 0; set < BATCH_SIZE % ds.nb_sets; set++)
        {
            double *input = ds.input[set];
            double *target = ds.target[set];
            error += train(net, input, target, lrate);
        }
        printf("epoch %04ld: error rate = %.5f (learning rate = %.3f)\n", i,
            error / BATCH_SIZE, lrate);

        lrate *= LRATE_MODIFIER;
    }
    printf("--------\n");

    return net;
}

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 4)
        errx(EXIT_FAILURE,
            "Usage: ./test {xor,logum,semeion,mnist} [path_to_saved_network] [test_accuracy]");

    network net;
    dataset ds;
    if (strcmp(argv[1], "semeion") == 0)
        ds = build_semeion(DS_SEMEION_FILE);
    else if (strcmp(argv[1], "mnist") == 0)
        ds = build_mnist(DS_MNIST_TRAIN_IMAGES, DS_MNIST_TRAIN_LABELS);
    else if (strcmp(argv[1], "logum") == 0)
        ds = build_logum(DS_LOGUM_FOLDER);
    else
        return train_on_xor();

    if (argc >= 3)
        net = load_network(argv[2]);
    else
    {
        net = train_on_ds(ds);
        save_network(net, "./saved.net");
    }

    if (argc == 4)
        test_accuracy(ds, net);
    else
        test_random_set(ds, net);

    free_dataset(ds);
    free_network(net);

    return EXIT_SUCCESS;
}
