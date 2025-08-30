#include <stdio.h>
#include <stdlib.h>

#include "networkStructure.h"

// Create network
// ACTIVATION CODES: 0 = sigmoid, 1 = relu, 2 = tanh, 3 = linear, 4 = softmax (only for output activation)
// activation functions must be an array of length hiddenLayers + 1. The input layer does not have an activation function
Network *createNetwork(int inputNeurons, int hiddenLayers, int *hiddenNeurons, int outputNeurons, int *activationFunctions) {
    // Check for errors
    if (inputNeurons < 1 || hiddenLayers < 0 || outputNeurons < 1) {
        handleError("invalid input to createNetwork.");
        return NULL;
    }
   
    // Create network
    Network *network = malloc(sizeof(Network));
    if (network == NULL) {
        handleError("failed to allocate network.");
        return NULL;
    }
    network->layerCount = hiddenLayers + 2;
    network->layers = malloc(network->layerCount * sizeof(Layer));
    if (network->layers == NULL) {
        freeNetwork(network);
        handleError("failed to allocate the layers of the network.");
        return NULL;
    }

    // Specify the activation functions for each layer
    // exclude input layer and exclude softmax from hidden layers
    network->activationFunctions = malloc((network->layerCount - 1) * sizeof(int));
    if (network->activationFunctions == NULL) {
        freeNetwork(network);
        handleError("failed to allocate the activation functions array.");
        return NULL;
    }
    for (int i = 0; i < network->layerCount - 1; i++) {
        if (i != network->layerCount - 2 && activationFunctions[i] == 4) {
            // softmax is only for the output layer
            freeNetwork(network);
            handleError("softmax can only be used in the output layer.");
            return NULL;
        }
        
        network->activationFunctions[i] = activationFunctions[i];
    }

    // Create input layer (no incoming weights for the input layer)
    if (initializeLayer(&network->layers[0], inputNeurons, 0) < 0) {
        freeNetwork(network);
        handleError("failed to allocate input layer.");
        return NULL;
    }
    
    // Set initial values
    for (int inputIndex = 0; inputIndex < inputNeurons; inputIndex++) {
        network->layers[0].biases[inputIndex] = ((float)rand() / RAND_MAX) * 0.6 - 0.3;
        network->layers[0].values[inputIndex] = 0;
        network->layers[0].biasGradients[inputIndex] = 0;
        network->layers[0].accumulatedBiasGradients[inputIndex] = 0;
    }



    // Create hidden layers
    for (int currentHiddenLayer = 1; currentHiddenLayer < hiddenLayers + 1; currentHiddenLayer++) {
        if (initializeLayer(&network->layers[currentHiddenLayer], hiddenNeurons[currentHiddenLayer - 1], network->layers[currentHiddenLayer - 1].neuronCount) < 0) {
            freeNetwork(network);
            handleError("failed to allocate hidden layer.");
            return NULL;
        }

        // Initialize hidden layer with random weights and biases between -0.3 and 0.3
        for (int currentHiddenNeuron = 0; currentHiddenNeuron < network->layers[currentHiddenLayer].neuronCount; currentHiddenNeuron++) {
            // assign random weights to each incoming weight and initialize the weight gradients and accumulated weight gradients
            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentHiddenLayer - 1].neuronCount; previousNeuronIndex++) {
                network->layers[currentHiddenLayer].incomingWeights[currentHiddenNeuron * network->layers[currentHiddenLayer - 1].neuronCount + previousNeuronIndex] = ((float)rand() / RAND_MAX) * 0.6 - 0.3; // Random weights between -0.3 and 0.3
                network->layers[currentHiddenLayer].weightGradients[currentHiddenNeuron * network->layers[currentHiddenLayer - 1].neuronCount + previousNeuronIndex] = 0;
                network->layers[currentHiddenLayer].accumulatedWeightGradients[currentHiddenNeuron * network->layers[currentHiddenLayer - 1].neuronCount + previousNeuronIndex] = 0;
            }
            
            // assign a random bias to the current neuron and set everything else to 0
            network->layers[currentHiddenLayer].biases[currentHiddenNeuron] = ((float)rand() / RAND_MAX) * 0.6 - 0.3;
            network->layers[currentHiddenLayer].values[currentHiddenNeuron] = 0;
            network->layers[currentHiddenLayer].biasGradients[currentHiddenNeuron] = 0;
            network->layers[currentHiddenLayer].accumulatedBiasGradients[currentHiddenNeuron] = 0;
        }
    }
    


    // Create output layer
    int outputLayerIndex = network->layerCount - 1;
    if (initializeLayer(&network->layers[outputLayerIndex], outputNeurons, network->layers[outputLayerIndex - 1].neuronCount) < 0) {
        freeNetwork(network);
        handleError("failed to allocate output layer.");
        return NULL;
    }

    
    // initialize output layer with random weights and biases between -0.3 and 0.3
    for (int currentOutputNeuron = 0; currentOutputNeuron < outputNeurons; currentOutputNeuron++) {
        // assign random weights to each incoming weight
        for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[outputLayerIndex - 1].neuronCount; previousNeuronIndex++) {
            // set the weights to a random value between -0.3 and 0.3
            network->layers[outputLayerIndex].incomingWeights[currentOutputNeuron * network->layers[outputLayerIndex - 1].neuronCount + previousNeuronIndex] = ((float)rand() / RAND_MAX) * 0.6 - 0.3;
            network->layers[outputLayerIndex].weightGradients[currentOutputNeuron * network->layers[outputLayerIndex - 1].neuronCount + previousNeuronIndex] = 0;
            network->layers[outputLayerIndex].accumulatedWeightGradients[currentOutputNeuron * network->layers[outputLayerIndex - 1].neuronCount + previousNeuronIndex] = 0;
        }
        
        // set the bias for the output node to random value between -0.3 and 0.3 and everything else to 0
        network->layers[outputLayerIndex].biases[currentOutputNeuron] = ((float)rand() / RAND_MAX) * 0.6 - 0.3;
        network->layers[outputLayerIndex].values[currentOutputNeuron] = 0;
        network->layers[outputLayerIndex].biasGradients[currentOutputNeuron] = 0;
        network->layers[outputLayerIndex].accumulatedBiasGradients[currentOutputNeuron] = 0;
    }

    return network;
}

int initializeLayer(Layer *layer, int neuronCount, int previousNeuronCount) {
    if (neuronCount == 0) {
        handleError("neuron count cannot be zero.");
        return ERR_UNEXPECTED_VAL;
    }

    // set neuron count of current layer
    layer->neuronCount = neuronCount;

    // allocate biases array
    layer->biases = malloc(neuronCount * sizeof(float));
    if (layer->biases == NULL) {
        handleError("failed to allocate biases array for layer.");
        return ERR_ALLOC;
    }

    // allocate values array
    layer->values = malloc(neuronCount * sizeof(float));
    if (layer->values == NULL) {
        handleError("failed to allocate values array for layer.");
        return ERR_ALLOC;
    }

    // allocate bias gradients array
    layer->biasGradients = malloc(neuronCount * sizeof(float));
    if (layer->biasGradients == NULL) {
        handleError("failed to allocate biasGradients array for layer.");
        return ERR_ALLOC;
    }

    // allocate error values array
    layer->errorValues = malloc(neuronCount * sizeof(float));
    if (layer->errorValues == NULL) {
        handleError("failed to allocate squared error values array for output layer.");
        return ERR_ALLOC;
    }

    // allocate incoming weights, weight gradients, and accumulated weight gradients if they exist, otherwise, set to NULL
    if (previousNeuronCount == 0) {
        layer->incomingWeights = NULL;
        layer->weightGradients = NULL;
        layer->accumulatedWeightGradients = NULL;
    } else {
        // allocate incoming weights array
        layer->incomingWeights = malloc(neuronCount * previousNeuronCount * sizeof(float));
        if (layer->incomingWeights == NULL) {
            handleError("failed to allocate incoming weights array for layer.");
            return ERR_ALLOC;
        }

        // allocate weight gradients array
        layer->weightGradients = malloc(neuronCount * previousNeuronCount * sizeof(float));
        if (layer->weightGradients == NULL) {
            handleError("failed to allocate weightGradients array for layer.");
            return ERR_ALLOC;
        }

        // allocate accumulated weight gradients array
        layer->accumulatedWeightGradients = malloc(neuronCount * previousNeuronCount * previousNeuronCount * sizeof(float));
        if (layer->accumulatedWeightGradients == NULL) {
            handleError("failed to allocate accumulatedWeightGradients array for layer.");
            return ERR_ALLOC;
        }
    }

    // allocate accumulated bias gradients array
    layer->accumulatedBiasGradients = malloc(neuronCount * sizeof(float));
    if (layer->accumulatedBiasGradients == NULL) {
        handleError("failed to allocate accumulated bias gradients array for layer.");
        return ERR_ALLOC;
    }

    // allocate weighted sums array
    layer->weightedSums = malloc(neuronCount * sizeof(float));
    if (layer->weightedSums == NULL) {
        handleError("failed to allocate weighted sums array for layer.");
        return ERR_ALLOC;
    }

    // allocate weighted errors array
    layer->weightedErrors = malloc(neuronCount * sizeof(float));
    if (layer->weightedErrors == NULL) {
        handleError("failed to allocate weighted errors array for layer.");
        return ERR_ALLOC;
    }

    // allocate activation function derivative array
    layer->activationFunctionDerivatives = malloc(neuronCount * sizeof(float));
    if (layer->activationFunctionDerivatives == NULL) {
        handleError("failed to allocate activation function derivatives array for layer.");
        return ERR_ALLOC;
    }

    return SUCCESS;
}

// free network
void freeNetwork(Network *network) {
    for (int currentLayer = 0; currentLayer < network->layerCount; currentLayer++) {
        if (network->layers[currentLayer].values != NULL) {
            free(network->layers[currentLayer].values);
            network->layers[currentLayer].values = NULL;
        }
        
        if (network->layers[currentLayer].biases != NULL) {
            free(network->layers[currentLayer].biases);
            network->layers[currentLayer].biases = NULL;
        }
        
        if (network->layers[currentLayer].weightGradients != NULL) {
            free(network->layers[currentLayer].weightGradients);
            network->layers[currentLayer].weightGradients = NULL;
        }

        if (network->layers[currentLayer].biasGradients != NULL) {
            free(network->layers[currentLayer].biasGradients);
            network->layers[currentLayer].biasGradients = NULL;
        }
        
        if (network->layers[currentLayer].incomingWeights != NULL) {
            free(network->layers[currentLayer].incomingWeights);
            network->layers[currentLayer].incomingWeights = NULL;
        }

        if (network->layers[currentLayer].accumulatedWeightGradients != NULL) {
            free(network->layers[currentLayer].accumulatedWeightGradients);
            network->layers[currentLayer].accumulatedWeightGradients = NULL;
        }

        if (network->layers[currentLayer].accumulatedBiasGradients != NULL) {
            free(network->layers[currentLayer].accumulatedBiasGradients);
            network->layers[currentLayer].accumulatedBiasGradients = NULL;
        }

        if (network->layers[currentLayer].errorValues != NULL) {
            free(network->layers[currentLayer].errorValues);
            network->layers[currentLayer].errorValues = NULL;
        }

        if (network->layers[currentLayer].weightedSums != NULL) {
            free(network->layers[currentLayer].weightedSums);
            network->layers[currentLayer].weightedSums = NULL;
        }

        if (network->layers[currentLayer].weightedErrors != NULL) {
            free(network->layers[currentLayer].weightedErrors);
            network->layers[currentLayer].weightedErrors = NULL;
        }

        if (network->layers[currentLayer].activationFunctionDerivatives != NULL) {
            free(network->layers[currentLayer].activationFunctionDerivatives);
            network->layers[currentLayer].activationFunctionDerivatives = NULL;
        }

        if (network->layers[currentLayer].momentumBiases != NULL) {
            free(network->layers[currentLayer].momentumBiases);
            network->layers[currentLayer].momentumBiases = NULL;
        }

        if (network->layers[currentLayer].momentumIncomingWeights != NULL) {
            free(network->layers[currentLayer].momentumIncomingWeights);
            network->layers[currentLayer].momentumIncomingWeights = NULL;
        }

        if (network->layers[currentLayer].rmsBiases != NULL) {
            free(network->layers[currentLayer].rmsBiases);
            network->layers[currentLayer].rmsBiases = NULL;
        }

        if (network->layers[currentLayer].rmsIncomingWeights != NULL) {
            free(network->layers[currentLayer].rmsIncomingWeights);
            network->layers[currentLayer].rmsIncomingWeights = NULL;
        }
    }
    if (network->activationFunctions != NULL) {
        free(network->activationFunctions);
    }
    if (network->layers != NULL) {
        free(network->layers);
    }
    if (network != NULL) {
        free(network);
    }
}