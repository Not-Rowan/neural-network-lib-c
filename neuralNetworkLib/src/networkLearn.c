#include <stdlib.h>

#include "networkLearn.h"

// feed forward by multiplying incoming value by incoming weights and adding the current neuron's bias
void feedForward(Network *network, float *input) {
    // check for invalid inputs
    if (!network || !network->layers || !network->activationFunctions) {
        handleError("null pointer in feedForward. Ensure network is properly initialized.");
        return;
    }

    if (!input) {
        handleError("invalid input to feedForward.");
        return;
    }

    // Set input layer values to input
    for (int inputNeuron = 0; inputNeuron < network->layers[0].neuronCount; inputNeuron++) {
        network->layers[0].values[inputNeuron] = input[inputNeuron];
    }

    // Calculate and set hidden layer and output layer values using the previous neuron's value multiplied by the neuron's incoming weights and add the current neuron's bias
    // Pass this into an activation function to get the final value
    for (int currentLayer = 1; currentLayer < network->layerCount; currentLayer++) {
        // init neuron counts for different layers
        int currentNeuronCount = network->layers[currentLayer].neuronCount;
        int previousNeuronCount = network->layers[currentLayer - 1].neuronCount;

        // calculate the weighted sum array (vector) for the current layer
        // (calculate the weighted sum for each neuron in the current layer)
        //
        // params: matrixA, matrixB, matrixC, rowsA, colsA, rowsB, colsB
        //
        // For value vector:
        // If rowsA = N and colsA = 1, the array is interpreted as a column vector (Nx1).
        // If rowsA = 1 and colsA = N, the array is interpreted as a row vector (1xN).
        // The 1D array's memory layout does not change, only how the function interprets it. In our case, it must be a column vector because the rows of matrix A must be equal to the columns of matrix B
        floatMatrixMultiply(network->layers[currentLayer].incomingWeights,   // matrixA
                            network->layers[currentLayer - 1].values,        // matrixB
                            network->layers[currentLayer].weightedSums,      // matrixC
                            currentNeuronCount, previousNeuronCount, previousNeuronCount, 1);

        // add biases to weighted sum
        for (int i = 0; i < currentNeuronCount; i++) {
            network->layers[currentLayer].weightedSums[i] += network->layers[currentLayer].biases[i];
        }

        // then apply the activation function to the weighted sum
        // the activation function is specified in the activationFunctions array
        switch (network->activationFunctions[currentLayer-1]) {
            case SIGMOID_ACTIVATION:
                // sigmoid
                sigmoid(network->layers[currentLayer].weightedSums, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            case RELU_ACTIVATION:
                // relu
                relu(network->layers[currentLayer].weightedSums, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            case TANH_ACTIVATION:
                // tanh
                tanhActivation(network->layers[currentLayer].weightedSums, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            case LINEAR_ACTIVATION:
                // linear
                linear(network->layers[currentLayer].weightedSums, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            case SOFTMAX_ACTIVATION:
                // softmax
                softmax(network->layers[currentLayer].weightedSums, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            default:
                // default to sigmoid
                sigmoid(network->layers[currentLayer].weightedSums, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
        }
    }
}



// create notes to explain neural networks and the backpropagation algorithm with all the math
// keep this with the library bc why not

// Function to compute the gradients for each parameter in the network
void computeGradients(Network *network, float *expectedOutputs) {
    // calculate the gradients for the output layer (the error)
    // the bias gradient alone is basically the error signal for the layer and multiplying that by each respective incoming value gives the gradient for the weights
    int outputLayerIndex = network->layerCount - 1;
    if (network->activationFunctions[outputLayerIndex-1] == SOFTMAX_ACTIVATION) {
        // softmax (softmax will automatically use cross entropy as the cost function)

        // cross entropy gradient is calculated as the difference between the target value and the predicted value for that output
        // predicted_value - target_value
        for (int outputNeuron = 0; outputNeuron < network->layers[outputLayerIndex].neuronCount; outputNeuron++) {
            network->layers[outputLayerIndex].biasGradients[outputNeuron] = network->layers[outputLayerIndex].values[outputNeuron] - expectedOutputs[outputNeuron];
            network->layers[outputLayerIndex].accumulatedBiasGradients[outputNeuron] += network->layers[outputLayerIndex].biasGradients[outputNeuron];

            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[outputLayerIndex-1].neuronCount; previousNeuronIndex++) {
                network->layers[outputLayerIndex].weightGradients[outputNeuron * network->layers[outputLayerIndex-1].neuronCount + previousNeuronIndex] = 
                    network->layers[outputLayerIndex].biasGradients[outputNeuron] * network->layers[outputLayerIndex - 1].values[previousNeuronIndex];
                network->layers[outputLayerIndex].accumulatedWeightGradients[outputNeuron * network->layers[outputLayerIndex-1].neuronCount + previousNeuronIndex] += 
                    network->layers[outputLayerIndex].weightGradients[outputNeuron * network->layers[outputLayerIndex-1].neuronCount + previousNeuronIndex];
            }
        }
    } else {
        // other activation functions (use squared error function)

        // first calculate the error of each neuron in the output layer
        squaredErrorDerivative(expectedOutputs, network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].squaredErrorValues, network->layers[outputLayerIndex].neuronCount);


        // then calculate the derivative of the activation function for each neuron in the output layer
        // this is stored in the gradients array for each neuron
        switch (network->activationFunctions[outputLayerIndex-1]) {
            case SIGMOID_ACTIVATION:
                // sigmoid
                sigmoidDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].activationFunctionDerivatives, network->layers[outputLayerIndex].neuronCount);
                break;
            case RELU_ACTIVATION:
                // relu
                reluDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].activationFunctionDerivatives, network->layers[outputLayerIndex].neuronCount);
                break;
            case TANH_ACTIVATION:
                // tanh
                tanhDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].activationFunctionDerivatives, network->layers[outputLayerIndex].neuronCount);
                break;
            case LINEAR_ACTIVATION:
                // linear
                linearDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].activationFunctionDerivatives, network->layers[outputLayerIndex].neuronCount);
                break;
            default:
                // default to sigmoid
                sigmoidDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].activationFunctionDerivatives, network->layers[outputLayerIndex].neuronCount);
                break;
        }

        // multiply the error by the derivative of the activation function for each neuron in the output layer and apply that to the gradient (this will be propogated backwards / applied later)
        for (int outputNeuron = 0; outputNeuron < network->layers[outputLayerIndex].neuronCount; outputNeuron++) {
            network->layers[outputLayerIndex].biasGradients[outputNeuron] = network->layers[outputLayerIndex].squaredErrorValues[outputNeuron] * network->layers[outputLayerIndex].activationFunctionDerivatives[outputNeuron];
            network->layers[outputLayerIndex].accumulatedBiasGradients[outputNeuron] += network->layers[outputLayerIndex].biasGradients[outputNeuron];

            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[outputLayerIndex].neuronCount; previousNeuronIndex++) {
                network->layers[outputLayerIndex].weightGradients[outputNeuron * network->layers[outputLayerIndex-1].neuronCount + previousNeuronIndex] = 
                    network->layers[outputLayerIndex].biasGradients[outputNeuron] * network->layers[outputLayerIndex - 1].values[previousNeuronIndex];
                network->layers[outputLayerIndex].accumulatedWeightGradients[outputNeuron * network->layers[outputLayerIndex-1].neuronCount + previousNeuronIndex] += 
                    network->layers[outputLayerIndex].weightGradients[outputNeuron * network->layers[outputLayerIndex-1].neuronCount + previousNeuronIndex];
            }
        }
    }


    // hidden layers
    for (int currentLayer = network->layerCount - 2; currentLayer > 0; currentLayer--) {
        // calculate the weighted error for the gradient
        // instead of transposing the weight matrix for the gradient calculation, i can just transpose the gradients vector and swap their places in the function to replicate the same thing.
        // instead of matA (transposed) * matB i can just swap matA and matB then transpose the new matA instead
        // Instead of this:
        //    matA        matB
        // [1, 2, 3]     [1]
        // [4, 5, 6]  x  [2]   (multiply rows of A by cols of B)
        // [7, 8, 9]     [3]
        //
        // I can do this:
        //   matB^T          matA
        //                 [1, 4, 7]
        // [1, 2, 3]   x   [2, 5, 8]  (multiply rows of B by cols of A)
        //                 [3, 6, 9]
        //
        floatMatrixMultiply(network->layers[currentLayer + 1].biasGradients,          // weighted gradients of the next layer (matrixA)
                            network->layers[currentLayer + 1].incomingWeights,          // weights connecting the current layer to the next layer (matrixB)
                            network->layers[currentLayer].weightedErrors,               // weighted error for the current layer (matrixC)
                            1, network->layers[currentLayer + 1].neuronCount, network->layers[currentLayer + 1].neuronCount, network->layers[currentLayer].neuronCount);

        // then calculate the derivative of the activation function for each neuron in the current layer
        switch (network->activationFunctions[currentLayer-1]) {
            case SIGMOID_ACTIVATION:
                // sigmoid
                sigmoidDerivative(network->layers[currentLayer].values, network->layers[currentLayer].activationFunctionDerivatives, network->layers[currentLayer].neuronCount);
                break;
            case RELU_ACTIVATION:
                // relu
                reluDerivative(network->layers[currentLayer].values, network->layers[currentLayer].activationFunctionDerivatives, network->layers[currentLayer].neuronCount);
                break;
            case TANH_ACTIVATION:
                // tanh
                tanhDerivative(network->layers[currentLayer].values, network->layers[currentLayer].activationFunctionDerivatives, network->layers[currentLayer].neuronCount);
                break;
            case LINEAR_ACTIVATION:
                // linear
                linearDerivative(network->layers[currentLayer].values, network->layers[currentLayer].activationFunctionDerivatives, network->layers[currentLayer].neuronCount);
                break;
            default:
                // default to sigmoid
                sigmoidDerivative(network->layers[currentLayer].values, network->layers[currentLayer].activationFunctionDerivatives, network->layers[currentLayer].neuronCount);
                break;
        }

        // then multiply the error by the derivative of the activation function for each neuron in the current layer and apply that to the gradient (this will be propogated backwards / applied later)
        // the bias gradient alone is basically the error signal for the layer and multiplying that by each respective incoming value gives the gradient for the weights
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            network->layers[currentLayer].biasGradients[currentNeuron] = network->layers[currentLayer].weightedErrors[currentNeuron] * network->layers[currentLayer].activationFunctionDerivatives[currentNeuron];
            network->layers[currentLayer].accumulatedBiasGradients[currentNeuron] += network->layers[currentLayer].biasGradients[currentNeuron];

            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer-1].neuronCount; previousNeuronIndex++) {
                network->layers[currentLayer].weightGradients[currentNeuron * network->layers[currentLayer-1].neuronCount + previousNeuronIndex] = 
                    network->layers[currentLayer].biasGradients[currentNeuron] * network->layers[currentLayer - 1].values[previousNeuronIndex];
                network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer-1].neuronCount + previousNeuronIndex] += 
                    network->layers[currentLayer].weightGradients[currentNeuron * network->layers[currentLayer-1].neuronCount + previousNeuronIndex];
            }
        }
    }
}

// Backpropagate using gradient descent.
// compute the gradients for each parameter in the network and apply the error to the network weights using gradient descent
void backPropagate(Network *network, float *expectedOutputs) {
    // compute the gradients for each layer
    computeGradients(network, expectedOutputs);

    // apply error to network weights using gradient descent
    SGDUpdate(network);

    // clear the gradients after applying them
    zeroGradients(network);
}

// Zero the gradients calculated using computeGradients
void zeroGradients(Network *network) {
    // clear the gradients & accumulatedGradients
    for (int currentLayer = 0; currentLayer < network->layerCount; currentLayer++) {
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            network->layers[currentLayer].biasGradients[currentNeuron] = 0;
            network->layers[currentLayer].accumulatedBiasGradients[currentNeuron] = 0;

            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer-1].neuronCount; previousNeuronIndex++) {
                network->layers[currentLayer].weightGradients[currentNeuron * network->layers[currentLayer-1].neuronCount + previousNeuronIndex] = 0;
                network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer-1].neuronCount + previousNeuronIndex] = 0;
            }
        }
    }
}

// average gradients across accumulated gradients for mini-batch gradient descent
void averageAccumulatedGradients(Network *network, int gradientCount) {
    // average the gradients based on gradientCount
    for (int currentLayer = 0; currentLayer < network->layerCount; currentLayer++) {
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            network->layers[currentLayer].accumulatedBiasGradients[currentNeuron] /= gradientCount;

            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer-1].neuronCount; previousNeuronIndex++) {
                network->layers[currentLayer].accumulatedWeightGradients[currentNeuron * network->layers[currentLayer-1].neuronCount + previousNeuronIndex] /= gradientCount;
            }
        }
    }
}