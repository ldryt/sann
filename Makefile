CFLAGS = -std=c99 -Wall -Wextra -Werror
LDFLAGS = -lm
CC = gcc
SRC = neural_network.c

xor:
	$(CC) -o xor.out $(SRC) xor.c $(CFLAGS) $(LDFLAGS)

clean:
	rm -f *.out
