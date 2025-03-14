#include <stdlib.h>
#include <math.h>

#include "optimizer.h"


// initialize the optimizer to default types using given parameters
// type: SGD (0), Momentum (1), RMSProp (2), Adam (3)
// IMPORTANT:
// change manually by hand by setting optimizer->parameter (before applying to network)
// network->optimizer->parameter (after applying to network)
Optimizer *initializeOptimizer(Network *network, int type, float learningRate) {
    if (network == NULL) {
        handleError("must create network before initalizing the optimizer.");
        return NULL;
    }
    
    // Initialize optimizer parameters w/ default values
    Optimizer *optimizer = malloc(sizeof(Optimizer));
    if (optimizer == NULL) {
        handleError("failed to allocate memory for the optimizer.");
    }
    
    optimizer->type = type;
    optimizer->learningRate = learningRate;
    optimizer->momentumCoefficient = 0.9;   // Default momentum coefficient or first moment decay rate
    optimizer->RMSPropDecay = 0.999;        // Default RMSProp decay rate or second moment decay rate 
    optimizer->epsilon = 1e-8;              // Small constant for numerical stability
    optimizer->timestep = 0;                // timestep for Adam bias correction


    // Initialize optimizer-specific variables in each layer

    // Input layer:

    // Momentum vars (or Adam)
    if (type == 1 || type == 3) {
        // weights
        network->layers[0].momentumIncomingWeights = NULL;

        // biases
        network->layers[0].momentumBiases = malloc(network->layers[0].neuronCount * sizeof(float));
        if (network->layers[0].momentumBiases == NULL) {
            freeNetwork(network);
            handleError("failed to initialize momentum biases for input layer.");
            return NULL;
        }

        // initialize momentum biases
        for (int inputIndex = 0; inputIndex < network->layers[0].neuronCount; inputIndex++) {
            network->layers[0].momentumBiases[inputIndex] = 0;
        }
    }
    
    // RMSProp vars (or Adam)
    if (type == 2 || type == 3) {
        // weights
        network->layers[0].rmsIncomingWeights = NULL;

        // biases
        network->layers[0].rmsBiases = malloc(network->layers[0].neuronCount * sizeof(float));
        if (network->layers[0].rmsBiases == NULL) {
            freeNetwork(network);
            handleError("failed to initialize RMSProp biases for input layer.");
            return NULL;
        }

        // initialize RMSProp biases
        for (int inputIndex = 0; inputIndex < network->layers[0].neuronCount; inputIndex++) {
            network->layers[0].rmsBiases[inputIndex] = 0;
        }
    }

    // Hidden layers:
    for (int hiddenLayerNum = 1; hiddenLayerNum < (network->layerCount-2) + 1; hiddenLayerNum++) {
        // Momentum vars (or Adam)
        if (type == 1 || type == 3) {
            // weights
            network->layers[hiddenLayerNum].momentumIncomingWeights = malloc(network->layers[hiddenLayerNum].neuronCount * network->layers[hiddenLayerNum-1].neuronCount * sizeof(float));
            if (network->layers[hiddenLayerNum].momentumIncomingWeights == NULL) {
                freeNetwork(network);
                handleError("failed to initialize momentum weights for hidden layer.");
                return NULL;
            }

            // biases
            network->layers[hiddenLayerNum].momentumBiases = malloc(network->layers[hiddenLayerNum].neuronCount * sizeof(float));
            if (network->layers[hiddenLayerNum].momentumBiases == NULL) {
                freeNetwork(network);
                handleError("failed to initialize momentum biases for hidden layer.");
                return NULL;
            }

            // initialize momentum weights and biases
            for (int currentNeuron = 0; currentNeuron < network->layers[hiddenLayerNum].neuronCount; currentNeuron++) {
                network->layers[hiddenLayerNum].momentumBiases[currentNeuron] = 0;
                for (int previousNeuron = 0; previousNeuron < network->layers[hiddenLayerNum-1].neuronCount; previousNeuron++) {
                    network->layers[hiddenLayerNum].momentumIncomingWeights[currentNeuron * network->layers[hiddenLayerNum-1].neuronCount + previousNeuron] = 0;
                }
            }
        }

        // RMSProp vars (or Adam)
        if (type == 2 || type == 3) {
            // weights
            network->layers[hiddenLayerNum].rmsIncomingWeights = malloc(network->layers[hiddenLayerNum].neuronCount * network->layers[hiddenLayerNum-1].neuronCount * sizeof(float));
            if (network->layers[hiddenLayerNum].rmsIncomingWeights == NULL) {
                freeNetwork(network);
                handleError("failed to initialize RMSProp weights for hidden layer.");
                return NULL;
            }

            // biases
            network->layers[hiddenLayerNum].rmsBiases = malloc(network->layers[hiddenLayerNum].neuronCount * sizeof(float));
            if (network->layers[hiddenLayerNum].rmsBiases == NULL) {
                freeNetwork(network);
                handleError("failed to initialize RMSProp biases for hidden layer.");
                return NULL;
            }

            // initialize RMSProp weights and biases
            for (int currentNeuron = 0; currentNeuron < network->layers[hiddenLayerNum].neuronCount; currentNeuron++) {
                network->layers[hiddenLayerNum].rmsBiases[currentNeuron] = 0;
                for (int previousNeuron = 0; previousNeuron < network->layers[hiddenLayerNum-1].neuronCount; previousNeuron++) {
                    network->layers[hiddenLayerNum].rmsIncomingWeights[currentNeuron * network->layers[hiddenLayerNum-1].neuronCount + previousNeuron] = 0;
                }
            }
        }
    }

    // for output layer
    int outputLayerIndex = network->layerCount - 1;

    // Momentum vars (or Adam)
    if (type == 1 || type == 3) {
        // weights
        network->layers[outputLayerIndex].momentumIncomingWeights = malloc(network->layers[outputLayerIndex].neuronCount * network->layers[outputLayerIndex-1].neuronCount * sizeof(float));
        if (network->layers[outputLayerIndex].momentumIncomingWeights == NULL) {
            freeNetwork(network);
            handleError("failed to initialize momentum weights for output layer.");
            return NULL;
        }

        // biases
        network->layers[outputLayerIndex].momentumBiases = malloc(network->layers[outputLayerIndex].neuronCount * sizeof(float));
        if (network->layers[outputLayerIndex].momentumBiases == NULL) {
            freeNetwork(network);
            handleError("failed to initialize momentum biases for output layer.");
            return NULL;
        }

        // initialize momentum weights and biases
        for (int currentNeuron = 0; currentNeuron < network->layers[outputLayerIndex].neuronCount; currentNeuron++) {
            network->layers[outputLayerIndex].momentumBiases[currentNeuron] = 0;
            for (int previousNeuron = 0; previousNeuron < network->layers[outputLayerIndex-1].neuronCount; previousNeuron++) {
                network->layers[outputLayerIndex].momentumIncomingWeights[currentNeuron * network->layers[outputLayerIndex-1].neuronCount + previousNeuron] = 0;
            }
        }
    }

    // RMSProp vars (or Adam)
    if (type == 2 || type == 3) {
        // weights
        network->layers[outputLayerIndex].rmsIncomingWeights = malloc(network->layers[outputLayerIndex].neuronCount * network->layers[outputLayerIndex-1].neuronCount * sizeof(float));
        if (network->layers[outputLayerIndex].rmsIncomingWeights == NULL) {
            freeNetwork(network);
            handleError("failed to initialize RMSProp weights for output layer.");
            return NULL;
        }

        // biases
        network->layers[outputLayerIndex].rmsBiases = malloc(network->layers[outputLayerIndex].neuronCount * sizeof(float));
        if (network->layers[outputLayerIndex].rmsBiases == NULL) {
            freeNetwork(network);
            handleError("failed to initialize RMSProp biases for output layer.");
            return NULL;
        }

        // initialize RMSProp weights and biases
        for (int currentNeuron = 0; currentNeuron < network->layers[outputLayerIndex].neuronCount; currentNeuron++) {
            network->layers[outputLayerIndex].rmsBiases[currentNeuron] = 0;
            for (int previousNeuron = 0; previousNeuron < network->layers[outputLayerIndex-1].neuronCount; previousNeuron++) {
                network->layers[outputLayerIndex].rmsIncomingWeights[currentNeuron * network->layers[outputLayerIndex-1].neuronCount + previousNeuron] = 0;
            }
        }
    }

    return optimizer;
}

// Function to apply the optimizer
void applyOptimizer(Network *network, Optimizer *optimizer) {
    network->optimizer = *optimizer;
}


// Function to update network using Stochastic Gradient Descent (SGD)
//
// Notation:
// θ (Theta)   = parameter (either weight or bias)
// α (Alpha)   = learning rate
// ∇L(θ) = gₜ   = gradient with respect to parameter θ
// β₁ (Beta 1) = momentum coefficient for SGD with momentum
// v           = velocity (accumulated gradient)
// β₂ (Beta 2) = decay factor for RMSProp
// E[g²]ₜ       = moving average of the squared gradients
// ε (Epsilon) = small constant to avoid division by zero
// η = α       = learning rate
// t           = timestep counter
//
// Update rule for SGD:
// θₜ₊₁ = θₜ - α∇L(θₜ)
//
// Update rule for SGD with momentum:
// vₜ = βvₜ₋₁ + ∇L(θₜ)  // Update the velocity term
// θₜ₊₁ = θₜ - αvₜ      // Update the parameter (weight or bias)
//
// Update rule for SGD RMSProp:
// E[g²]ₜ = β₂ * E[g²]ₜ₋₁ + (1 - β₂) * gₜ²  // Update the moving average of the squared gradients.
// θₜ₊₁ = θₜ - (η / √(E[g²]ₜ + ε)) * gₜ     // Update the parameter (weight or bias)
//
// Update rule for Adam:
// gₜ = ∇L(θₜ)                      // compute gradient
// mₜ = β₁mₜ₋₁ + (1 - β₁)gₜ         // Update biased first moment (momentum term)
// vₜ = β₂vₜ₋₁ + (1 - β₂)gₜ²        // Update biased second moment (RMSProp term)
// mₜ = mₜ / (1 - β₁^t)             // Correct bias in the first moment
// vₜ = vₜ / (1 - β₂^t)             // Correct bias in the second moment
// θₜ₊₁ = θₜ - (α / (√vₜ + ε))mₜ    // Update the parameter (weight or bias)
//

void SGDUpdate(Network *network) {
    for (int currentLayer = network->layerCount - 1; currentLayer > 0; currentLayer--) {
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            // formatted like this for optimization reasons
            if (network->optimizer.type == SGD_MOMENTUM) {
                // if optimizer is momentum
                network->layers[currentLayer].momentumBiases[currentNeuron] = network->optimizer.momentumCoefficient * network->layers[currentLayer].momentumBiases[currentNeuron] + network->layers[currentLayer].accumulatedBiasGradients[currentNeuron];
                network->layers[currentLayer].biases[currentNeuron] -= network->optimizer.learningRate * network->layers[currentLayer].momentumBiases[currentNeuron];

                for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                    network->layers[currentLayer].momentumIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] = network->optimizer.momentumCoefficient * network->layers[currentLayer].momentumIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] + network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex];
                    network->layers[currentLayer].incomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] -= network->optimizer.learningRate * network->layers[currentLayer].momentumIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex];
                }
            } else if (network->optimizer.type == SGD_RMS_PROP) {
                // if optimizer is RMSProp
                network->layers[currentLayer].rmsBiases[currentNeuron] = network->optimizer.RMSPropDecay * network->layers[currentLayer].rmsBiases[currentNeuron] + (1 - network->optimizer.RMSPropDecay) * (network->layers[currentLayer].accumulatedBiasGradients[currentNeuron] * network->layers[currentLayer].accumulatedBiasGradients[currentNeuron]);
                network->layers[currentLayer].biases[currentNeuron] -= network->optimizer.learningRate * network->layers[currentLayer].accumulatedBiasGradients[currentNeuron] / (sqrt(network->layers[currentLayer].rmsBiases[currentNeuron]) + network->optimizer.epsilon);

                for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                    network->layers[currentLayer].rmsIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] = network->optimizer.RMSPropDecay * network->layers[currentLayer].rmsIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] + (1 - network->optimizer.RMSPropDecay) * (network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] * network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex]);
                    network->layers[currentLayer].incomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] -= network->optimizer.learningRate * (network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex]) / (sqrt(network->layers[currentLayer].rmsIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex]) + network->optimizer.epsilon);
                }
            } else if (network->optimizer.type == SGD_ADAM) {
                // if optimizer is Adam

                // increment the step counter
                // technically this is only supposed to be updated every mini batch or single gradient descent step but it's okay for now?
                network->optimizer.timestep++;

                // Compute first moment estimate then compute the bias-corrected first moment estimate (momentum movement)
                network->layers[currentLayer].momentumBiases[currentNeuron] = network->optimizer.momentumCoefficient * network->layers[currentLayer].momentumBiases[currentNeuron] + (1 - network->optimizer.momentumCoefficient) * network->layers[currentLayer].accumulatedBiasGradients[currentNeuron];
                network->layers[currentLayer].momentumBiases[currentNeuron] = network->layers[currentLayer].momentumBiases[currentNeuron] / (1 - pow(network->optimizer.momentumCoefficient, network->optimizer.timestep));

                for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                    network->layers[currentLayer].momentumIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] = network->optimizer.momentumCoefficient * network->layers[currentLayer].momentumIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] + (1 - network->optimizer.momentumCoefficient) * network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex];
                    network->layers[currentLayer].momentumIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] = network->layers[currentLayer].momentumIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] / (1 - pow(network->optimizer.momentumCoefficient, network->optimizer.timestep));
                }

                // Compute second raw momentum estimate then compute the bias-corrected second moment estimate (RMSProp movement)
                network->layers[currentLayer].rmsBiases[currentNeuron] = network->optimizer.RMSPropDecay * network->layers[currentLayer].rmsBiases[currentNeuron] + (1 - network->optimizer.RMSPropDecay) * (network->layers[currentLayer].accumulatedBiasGradients[currentNeuron] * network->layers[currentLayer].accumulatedBiasGradients[currentNeuron]);
                network->layers[currentLayer].rmsBiases[currentNeuron] = network->layers[currentLayer].rmsBiases[currentNeuron] / (1 - pow(network->optimizer.RMSPropDecay, network->optimizer.timestep));

                for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                    network->layers[currentLayer].rmsIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] = network->optimizer.RMSPropDecay * network->layers[currentLayer].rmsIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] + (1 - network->optimizer.RMSPropDecay) * (network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] * network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex]);
                    network->layers[currentLayer].rmsIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] = network->layers[currentLayer].rmsIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] / (1 - pow(network->optimizer.RMSPropDecay, network->optimizer.timestep));
                }

                // Update parameters
                network->layers[currentLayer].biases[currentNeuron] -= network->optimizer.learningRate * network->layers[currentLayer].momentumBiases[currentNeuron] / (sqrt(network->layers[currentLayer].rmsBiases[currentNeuron]) + network->optimizer.epsilon);
            
                for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                    network->layers[currentLayer].incomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] -=  network->optimizer.learningRate * network->layers[currentLayer].momentumIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] / (sqrt(network->layers[currentLayer].rmsIncomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex]) + network->optimizer.epsilon);
                }
            } else {
                // if optimizer is just SGD (also default)
                network->layers[currentLayer].biases[currentNeuron] -= network->layers[currentLayer].accumulatedBiasGradients[currentNeuron] * network->optimizer.learningRate;
            
                for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                    network->layers[currentLayer].incomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] -= network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex] * network->optimizer.learningRate;
                }
            }
        }
    }
}