#ifndef NETWORKSTRUCTURE_H
#define NETWORKSTRUCTURE_H

#include "networkErrors.h"

// Declare network structures

// Optimizer structure to contain optimizer-specific variables
typedef struct Optimizer {
    int type;                   // 0: SGD, 1: Momentum, 2: RMSProp, 3: Adam
    float learningRate;         // Base learning rate
    float momentumCoefficient;  // Momentum coefficient (Momentum) or first moment decay rate (Adam)
    float RMSPropDecay;         // Decay rate for RMSProp (RMSProp) or second moment decay rate (ADAM)
    float epsilon;              // Small constant for numerical stability
    size_t timestep;            // Step count for bias correction (Adam-specific)
} Optimizer;

// Layer structure to organize and contain variables contained in each layer
typedef struct Layer {
    float *incomingWeights;                   // weights from previous layer to this layer. Note: This is a 2D array but for memory and matrix multiplication purposes, this is going to be defined as a 1D array.
    float *biases;                            // biases for each neuron in this layer
    float *values;                            // values of each neuron in this layer
    float *weightGradients;                  // gradients of each weight in this layer (temporary and added to accumulated weight gradients to be applied to each parameter)
    float *biasGradients;                     // gradients of each bias in this layer (temporary and added to accumulated bias gradients to be applied to each parameter)
    float *accumulatedWeightGradients;       // stores the accumulated weight gradients
    float *accumulatedBiasGradients;          // stores the accumulated bias gradients
    float *weightedSums;                      // used to store the weighted sum during the forward pass step
    float *weightedErrors;                    // used to store the weighted errors (resulting matrix after multiplying the incoming weights by the gradients)
    float *errorValues;                       // used to store the error values in the output layer
    float *activationFunctionDerivatives;     // used to store the derivatives of the activation functions
    int neuronCount;                          // number of neurons in this layer
    
    // Optimizer-specific arrays
    float *momentumIncomingWeights; // 2D array for velocity with respect to the weights. Same note as incomingWeights
    float *momentumBiases;          // Momentum for biases
    float *rmsIncomingWeights;      // RMS for weights (RMSProp, Adam)
    float *rmsBiases;               // RMS for biases
} Layer;

// Higher level network structure to contain layers, optimizer settings, etc
typedef struct Network {
    Layer *layers;              // each layer of the network has its own weights, biases, values, and gradients
    int *activationFunctions;   // each layer has its own activation function
    int layerCount;             // number of layers in the network
    Optimizer optimizer;        // optimizer configuration
} Network;

// Redeclare functions

// Creates a neural network with the proper structure
// parameters:
//      inputNeurons: the number of input neurons that the network will have
//      hiddenLayers: the number of hidden layers that the network will have
//      hiddenNeurons: an array where each index holds the number of neurons for the respective hidden layer
//      outputNeurons: the number of output neurons that the network will have
//      activationFunctions: an array defining the activation function for each layer, index 0 starting at the first hidden layer and going until the output layer. Uses the activation codes defined in "activationFunction.h".
Network *createNetwork(int inputNeurons, int hiddenLayers, int *hiddenNeurons, int outputNeurons, int *activationFunctions);

// frees the memory of the network
void freeNetwork(Network *network);

// helper function for the createNetwork() function
// parameters:
//      layer: specifies the layer to initialize
//      neuronCount: the number of neurons located in the specified layer
//      previousNeuronCount: the number of neurons in the previous layer (0 if input layer)
int initializeLayer(Layer *layer, int neuronCount, int previousNeuronCount);

#endif