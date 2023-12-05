#include "neural_network.h"
#include "utils/semeion_utils.h"

#define EPOCHS          300
#define NB_INPUTS       256 // 16x16
#define NB_HIDDEN       32
#define NB_OUTPUTS      10
#define LEARNING_RATE   1.66
#define LRATE_MODIFIER  0.999

// Prints an array
void print_array(double* array, int size)
{
    for (int i = 0; i < size; i++)
        printf("%f ", (double) array[i]);
    printf("\n");
}

int main()
{
    // srand(time(0));
    dataset ds = build("./datasets/semeion/semeion.data", NB_INPUTS, NB_OUTPUTS);
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

    double* input = ds.input[0];
    double* target = ds.target[0];
    double* prediction = feed(net, input);

    printf("\n----\n\n");
    printf("Target:\n");
    print_array(target, NB_OUTPUTS);
    printf("Prediction:\n");
    print_array(prediction, NB_OUTPUTS);

    free_dataset(ds);
    free_network(net);

    return EXIT_SUCCESS;
}
