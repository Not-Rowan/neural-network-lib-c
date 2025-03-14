#ifndef OPTIMISER_H
#define OPTIMISER_H

#include "networkStructure.h"

// define optimizer types
#define SGD 0
#define SGD_MOMENTUM 1
#define SGD_RMS_PROP 2
#define SGD_ADAM 3

// Redefine functions
void SGDUpdate(Network *network);
Optimizer *initializeOptimizer(Network *network, int type, float learningRate);

// basically just sets the optimizer struct to the one returned by initializeOptimizer()
void applyOptimizer(Network *network, Optimizer *optimizer);

#endif