CFLAGS = -std=c99 -Wall -Wextra
LDFLAGS = -lm
CC = gcc
SRC = neural_network.c math_utils.c

all: xor semeion

xor:
	$(CC) xor.c $(SRC) $(CFLAGS) $(LDFLAGS) -o xor.out
semeion:
	$(CC) semeion.c semeion_utils.c $(SRC) $(CFLAGS) $(LDFLAGS) -o semeion.out

clean:
	rm -f *.out *.o
