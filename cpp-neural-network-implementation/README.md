# Neural Network Using C++
This project is an implementation of a simple feed forward neural network using C++.


## Description
It consists of a Perceptron Class and a Multilayer Perceptron Class.


## Requirements
- C++17 compiler support
- CMake 

This work is inspired by the example files and the coursework from the LinkedIn Learning course titled
> Training a Neural Network in C++

Future updates may include implementations of matrix and linear algebra operations or use of some already available libraries like DLib and Caffe.


(This might be ported to a separate repositary depending on the scale)


The `NeuralNetworkTester.cpp` and resulting executable tests the neural network developed. It also contains a small program to test a neural network for recognising digits from a seven segment display.
Three network architectures were used:
1. 7-7-1 : The output neuron gives a value between 0-9. This network is not ideal and the neural net learn some unnecessary information.
2. 7-7-10: Each output neuron is a binary classifier and the neuron with the highest probability is the predicted digit. This is ideal.
3. 7-7-7: This architecture can provide insight on how the neural network works.

#### TODO:
The Segment Digit Recognition program would be moved to another source file. 
A GUI using JUCE would be also made to ease entering the segment brightness levels.


