#ifndef NETWORKLEARN_H
#define NETWORKLEARN_H

#include "networkStructure.h"
#include "activationFunctions.h"
#include "errorFunctions.h"
#include "optimizer.h"
#include "linearAlgebra.h"

// Redefine functions

// feeds an input vector forwards through a network
// the parameters are the network structure and the input array to the network
void feedForward(Network *network, float *input);

// computes and stores the gradients for the network given an array of expected outputs.
// Note: gradients computed from this function are added to the network's internal gradient storage. ensure you call zeroGradients() to reset these gradients
// the parameters are the network structure and the array of expected outputs
void computeGradients(Network *network, float *expectedOutputs);

// zeros the gradients in the network
void zeroGradients(Network *network);

// averages the accumulated gradients from computeGradients()
// gradientCount is the amount of gradients accumulated since zeroGradients was called
void averageAccumulatedGradients(Network *network, int gradientCount);

// calculates gradients, propagates, and applies the stored gradients backwards throughout the network
// this function is basically just stochastic gradient descent
void backPropagate(Network *network, float *expectedOutputs);


// computes weighted sums for a given layer
void computeWeightedSums(Network *network, int layer);

// computes weighted errors for a given layer
void computeWeightedErrors(Network *network, int layer);

// helper functions to compute layer gradients
void computeLayerGradients(Network *network, int layer, float *error);
void updateBiasGradient(Network *network, int layer, int neuron, float error);
void updateWeightGradient(Network *network, int layer, int neuron, float error);

#endif