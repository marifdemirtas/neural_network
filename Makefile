CC=g++

main: main.o NN.o NN.h
	$(CC) NN.o main.o -o main -O2 -larmadillo

test: tests/test
	tests/test

tests/test: NN.o tests/test.o
	rm -f tests/test && g++ tests/test.o NN.o -o tests/test -O2 -larmadillo

NN.o: NN.cpp NN.h
	$(CC) NN.cpp -c -o NN.o -O2

main.o: main.cpp NN.h
	$(CC) main.cpp -c -o main.o -O2

tests/test.o: NN.h tests/test.cpp
	g++ tests/test.cpp -c -o tests/test.o

clean:
	rm -f main.o NN.o main && rm -f make_test/*

