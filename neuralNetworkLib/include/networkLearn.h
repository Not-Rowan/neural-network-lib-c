#ifndef NETWORKLEARN_H
#define NETWORKLEARN_H

#include "networkStructure.h"
#include "activationFunctions.h"
#include "errorFunctions.h"
#include "optimizer.h"
#include "linearAlgebra.h"




// feedForward() passes an input vector fully through the network (forward pass)
// Parameters:
//     - network: pointer to the network
//     - input: input vector containing values to pass through the network
// Return:
//     - Nothing
void feedForward(Network *network, float *input);

// computeGradients() computes and stores the gradients for the network given a vector of expected outputs
// Note: gradients computed from this function are added to the network's internal gradient storage. ensure you call zeroGradients() to reset these gradients
// Parameters:
//     - network: pointer to the network
//     - expectedOutputs: vector of expected values
// Return:
//     - Nothing
void computeGradients(Network *network, float *expectedOutputs);

// zeroGradients() zeros the gradients of the network (stored in the network struct)
// Parameters:
//     - network: pointer to the network
// Return:
//     - Nothing
void zeroGradients(Network *network);

// averageAccumulatedGradients() averages the gradients accumulated from computeGradients()
// Parameters:
//     - network: pointer to the network
//     - gradientCount: the amount of gradients accumulated since zeroGradients was called
// Return:
//     - Nothing
void averageAccumulatedGradients(Network *network, int gradientCount);

// backPropagate() computes gradients, updates parameters, and zeroes the gradients
// Can just be used after feedForward() if complete control of the backpropatation process is not necessary
// Parameters:
//     - network: pointer to the network
//     - expectedOutputs: vector of expected values
// Return:
//     - Nothing
void backPropagate(Network *network, float *expectedOutputs);


// computeWeightedSums() is a helper function for feedForward() that computes the weighted sums for a given layer
// Parameters:
//     - network: pointer to the network
//     - layer: layer number
// Return:
//     - Nothing
void computeWeightedSums(Network *network, int layer);

// computeWeightedErrors() is a helper function for computeGradients() that computes the weighted gradients or weighted error for a given layer 
// Parameters:
//     - network: pointer to the network
//     - layer: layer number
// Return:
//     - Nothing
void computeWeightedErrors(Network *network, int layer);

// computeLayerGradients() is a helper function for computeGradients() that computes the gradients for a given layer
// Parameters:
//     - network: pointer to the network
//     - layer: layer number
//     - error: pointer to the error vector for the current layer
// Return:
//     - Nothing
void computeLayerGradients(Network *network, int layer, float *error);

// updateBiasGradient() is a helper function for computeLayerGradients() that updates the bias gradient within the network struct
// Parameters:
//     - network: pointer to the network
//     - layer: layer number
//     - neuron: neuron number for the layer
//     - error: error value for the current neuron
// Return:
//     - Nothing
void updateBiasGradient(Network *network, int layer, int neuron, float error);

// updateBiasGradient() is a helper function for computeLayerGradients() that updates the weight gradient within the network struct
// Parameters:
//     - network: pointer to the network
//     - layer: layer number
//     - neuron: neuron number for the layer
//     - error: error value for the current neuron
// Return:
//     - Nothing
void updateWeightGradient(Network *network, int layer, int neuron, float error);

#endif