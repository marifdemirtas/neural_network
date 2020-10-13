CC=g++

main: main.o NN.o
	$(CC) NN.o main.o -o main -larmadillo

NN.o: NN.cpp
	$(CC) NN.cpp -c -o NN.o

main.o: main.cpp
	$(CC) main.cpp -c -o main.o

clean:
	rm -f main.o NN.o main && rm make_test/*

run_verbose:
	rm make_test/* && ./main files/input6.txt files/set6.txt make_test

test: NN.cpp tests/test.cpp
	rm -f tests/test && g++ tests/test.cpp NN.cpp -o tests/test -larmadillo && tests/test 
