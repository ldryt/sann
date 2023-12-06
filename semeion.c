#include "neural_network.h"
#include "utils/semeion_utils.h"

#define EPOCHS          30
#define NB_INPUTS       256 // 16x16
#define NB_HIDDEN       32
#define NB_OUTPUTS      10
#define LEARNING_RATE   1.33
#define LRATE_MODIFIER  0.996

network new_network(dataset ds)
{
    network net = init_network(NB_INPUTS, NB_HIDDEN, NB_OUTPUTS);

    double lrate = LEARNING_RATE;
    for (int i = 0; i < EPOCHS; i++)
    {
        shuffle(ds);

        double error = 0;
        for (size_t row = 0; row < ds.nb_rows; row++)
        {
            double* input = ds.input[row];
            double* target = ds.target[row];
            error += train(net, input, target, lrate);
        }
        printf("epoch %04d: error rate = %.5f | learning rate = %.2f\n",
            i, (double) error / ds.nb_rows, (double) lrate);

        lrate *= LRATE_MODIFIER;
    }
    printf("\n----\n\n");

    return net;
}

int main(int argc, char *argv[])
{
    if (argc > 2)
        errx(EXIT_FAILURE, "Only one path can be passed");
    
    network net;
    dataset ds = build("./datasets/semeion/semeion.data", NB_INPUTS, NB_OUTPUTS);
    if (argc == 1)
        net = new_network(ds);
    else
        net = load_network(argv[1]);
    
    shuffle(ds);

    double* input = ds.input[0];
    double* target = ds.target[0];
    double* prediction = feed(net, input);

    printf("Target:\n");
    print_array(target, NB_OUTPUTS);
    printf("Prediction:\n");
    print_array(prediction, NB_OUTPUTS);

    save_network(net, "./semeion.net");

    free_dataset(ds);
    free_network(net);

    return EXIT_SUCCESS;
}
