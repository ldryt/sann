CFLAGS = -std=c99 -Wall -Wextra -Werror
LDFLAGS = -lm
CC = gcc
SRC = neural_network.c

all: xor

xor:
	$(CC) -o xor.out $(SRC) xor.c $(CFLAGS) $(LDFLAGS)
numbers:
	$(CC) -o numbers.out $(SRC) numbers.c $(CFLAGS) $(LDFLAGS)

clean:
	rm -f *.out
