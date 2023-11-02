#include "neural_network.h"

#define EPOCHS 100000
#define NB_INPUTS 2
#define NB_HIDDEN_NEURONS 4
#define NB_OUTPUTS 1
#define LEARNING_RATE 1.0

int main(void)
{
    double inputs[4][NB_INPUTS] = {{ 0, 0 },{ 0, 1 },{ 1, 0 },{ 1, 1 }};
    double expected_output[4] = { 0, 1, 1, 0 };
    network net = init_network(NB_INPUTS, NB_HIDDEN_NEURONS, NB_OUTPUTS);
    printf("\nError rate:\n---\n");
    for(size_t i = 0; i < EPOCHS; i++)
    {
        double error = 0;
        for(size_t j = 0; j < 4; j++)
        {   
            error = train(net, inputs[j], &expected_output[j], LEARNING_RATE);
        }
        if (i%7500==0) printf("Epoch %5ld: %f\n", i, error);
    }

    printf("\nResults:\n---\n");
    for(size_t i = 0; i < 4; i++)
    {   
        double* prediction = feed(net, inputs[i]);
        
        printf("{%d,%d} --> %g", (int)inputs[i][0], (int)inputs[i][1], (double)prediction[0]);
        printf("\n");
    }

    return EXIT_SUCCESS;
}
