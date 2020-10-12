CC=g++

main: main.o NN.o
	$(CC) NN.o main.o -o main -larmadillo

NN.o: NN.cpp
	$(CC) NN.cpp -c -o NN.o

main.o: main.cpp
	$(CC) main.cpp -c -o main.o

clean:
	rm main.o NN.o main