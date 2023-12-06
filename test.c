#include "neural_network.h"
#include "utils/semeion_utils.h"
#include "utils/mnist_utils.h"

#define EPOCHS                  30
#define HIDDEN_NEURONS          32
#define LEARNING_RATE           1.33
#define LRATE_MODIFIER          0.996

#define DS_MNIST_TRAIN_IMAGES   "./datasets/mnist/train-images.idx3-ubyte"
#define DS_MNIST_TRAIN_LABELS   "./datasets/mnist/train-labels.idx1-ubyte"
#define DS_SEMEION_FILE         "./datasets/semeion/semeion.data"

int xor_network()
{
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double expected_output[4] = {0, 1, 1, 0};
    network net = init_network(2, 4, 1);
    printf("\nError rate:\n---\n");
    double error = 0;
    for (size_t i = 0; i < EPOCHS * 1000; i++)
    {
        for (size_t j = 0; j < 4; j++)
            error = train(net, inputs[j], &expected_output[j], LEARNING_RATE);
        if (i % 1000 == 0)
            printf("Epoch %5ld: %f\n", i, error);
    }

    printf("\nResults:\n---\n");
    for (size_t i = 0; i < 4; i++)
    {
        double *prediction = feed(net, inputs[i]);

        printf("{%d,%d} --> %g", (int)inputs[i][0], (int)inputs[i][1],
            (double)prediction[0]);
        printf("\n");
    }

    return EXIT_SUCCESS;
}

network train_network(dataset ds)
{
    network net = init_network(ds.nb_inputs, HIDDEN_NEURONS, ds.nb_outputs);

    double lrate = LEARNING_RATE;
    for (int i = 0; i < EPOCHS; i++)
    {
        shuffle(ds);

        double error = 0;
        for (size_t row = 0; row < ds.nb_sets; row++)
        {
            double* input = ds.input[row];
            double* target = ds.target[row];
            error += train(net, input, target, lrate);
        }
        printf("epoch %04d: error rate = %.5f | learning rate = %.2f\n",
            i, (double) error / ds.nb_sets, (double) lrate);

        lrate *= LRATE_MODIFIER;
    }
    printf("\n--------\n");

    return net;
}

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3)
        errx(EXIT_FAILURE, "Usage: ./test {xor,semeion,mnist} [path_to_saved_network]");

    network net;
    dataset ds;
    if (strcmp(argv[1], "semeion") == 0)
        ds = build_semeion(DS_SEMEION_FILE);
    else if (strcmp(argv[1], "mnist") == 0)
        return 1;
        //ds = build_mnist(DS_MNIST_TRAIN_IMAGES, DS_MNIST_TRAIN_LABELS);
    else
        return xor_network();

    if (argc == 3)
        net = load_network(argv[2]);
    else
        net = train_network(ds);
    
    srand(time(NULL));
    shuffle(ds);

    double* input = ds.input[0];
    double* target = ds.target[0];
    double* prediction = feed(net, input);

    printf("Target:\n");
    print_array(target, net.nb_outputs);
    printf("Prediction:\n");
    print_array(prediction, net.nb_outputs);

    save_network(net, "./saved.net");

    free_dataset(ds);
    free_network(net);

    return EXIT_SUCCESS;
}
