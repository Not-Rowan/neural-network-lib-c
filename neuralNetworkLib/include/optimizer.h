#ifndef OPTIMISER_H
#define OPTIMISER_H

#include "networkStructure.h"

// define optimizer types
#define SGD 0
#define SGD_MOMENTUM 1
#define SGD_RMS_PROP 2
#define SGD_ADAM 3


// SGDUpdate() updates the network parameters based on stored gradients
// Parameters:
//     - network: pointer to the network
// Return:
//     - Nothing
void SGDUpdate(Network *network);

// initializeOptimizer() initializes the optimizer struct based on the given parameters
// Parameters:
//     - network: pointer to the network
//     - type: type of optimizer (e.g. SGD, Adam, RMSProp)
//     - learningRate: the learning rate of the network
// Return:
//     - pointer to the new optimizer struct
Optimizer *initializeOptimizer(Network *network, int type, float learningRate);

// applyOptimizer() applies the optimizer returned by initializeOptimizer()
// Parameters:
//     - network: pointer to the network
//     - optimizer: pointer to the optimizer struct
// Return:
//     - Nothing
void applyOptimizer(Network *network, Optimizer *optimizer);

#endif