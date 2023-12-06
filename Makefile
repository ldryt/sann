CFLAGS = -std=c99 -Wall -Wextra
LDFLAGS = -lm
CC = gcc
SRC = neural_network.c utils/math_utils.c utils/dataset_utils.c utils/semeion_utils.c utils/mnist_utils.c

all: test

test:
	$(CC) test.c $(SRC) $(CFLAGS) $(LDFLAGS) -o test

clean:
	$(RM) test
