cmake_minimum_required(VERSION 3.10)

project("CPP Neural Network" VERSION 1.0 DESCRIPTION "Implementation of Neural Network in C++")



# set compiler and flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES MLP.cpp NeuralNetworkTester)


add_executable(NeuralNetworkTester ${SOURCES})
