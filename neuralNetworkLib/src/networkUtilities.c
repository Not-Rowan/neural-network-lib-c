#include <stdio.h>
#include <stdlib.h>

#include "networkUtilities.h"

// print the structure of the network including layer num, node num, node values, node biases, and incoming weights
void printNetwork(Network *network) {
    printf("---Network---\n");
    for (int currentLayer = 0; currentLayer < network->layerCount; currentLayer++) {
        printf("Layer %d", currentLayer);
        if (currentLayer == 0) {
            printf(" (input):\n");
        } else if (currentLayer == network->layerCount - 1) {
            printf(" (output):\n");
        } else {
            printf(" (hidden):\n");
        }

        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            printf("    Node %d:\n        Value: %f\n        Bias: %f\n", currentNeuron, network->layers[currentLayer].values[currentNeuron], network->layers[currentLayer].biases[currentNeuron]);
            // skip printing weights if input layer because the input layer has no incoming weights
            if (currentLayer == 0) continue;
            printf("        Incoming Weights: {");
            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                printf("%f", network->layers[currentLayer].incomingWeights[currentNeuron * network->layers[currentLayer - 1].neuronCount + previousNeuronIndex]);
                if (previousNeuronIndex == network->layers[currentLayer - 1].neuronCount - 1) continue;
                printf(", ");
            }
            printf("}\n");
        }
    }
}

// Function to copy one network to another as long as the structure is the same
void copyNetwork(Network *destination, Network *source) {
    // check if the layer counts are the same
    if (destination->layerCount != source->layerCount) {
        return;
    }

    // copy the biases, values, gradients, and incoming weights
    for (int currentLayer = 0; currentLayer < source->layerCount; currentLayer++) {
        // check if the neuron counts are the same
        if (destination->layers[currentLayer].neuronCount != source->layers[currentLayer].neuronCount) {
            return;
        }

        // copy the biases and incoming weights
        for (int currentNeuron = 0; currentNeuron < source->layers[currentLayer].neuronCount; currentNeuron++) {
            destination->layers[currentLayer].biases[currentNeuron] = source->layers[currentLayer].biases[currentNeuron];
            if (currentLayer != 0) {
                for (int previousNeuronIndex = 0; previousNeuronIndex < source->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                    destination->layers[currentLayer].incomingWeights[currentNeuron * source->layers[currentLayer - 1].neuronCount + previousNeuronIndex] = source->layers[currentLayer].incomingWeights[currentNeuron * source->layers[currentLayer - 1].neuronCount + previousNeuronIndex];
                }
            }
        }
    }
}

// Export network for loading later (.json file format)
void exportNetworkJSON(Network *network, char *filename) {
    // open file
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        return;
    }

    // write to file
    fprintf(fp, "{\n");
    fprintf(fp, "    \"layerCount\": %d,\n", network->layerCount);
    fprintf(fp, "    \"activationFunctions\": [\n");
    for (int i = 0; i < network->layerCount - 1; i++) {
        fprintf(fp, "        %d", network->activationFunctions[i]);
        if (i == network->layerCount - 2) {
            fprintf(fp, "\n");
        } else {
            fprintf(fp, ",\n");
        }
    }
    fprintf(fp, "    ],\n");
    fprintf(fp, "    \"layers\": [\n");
    for (int i = 0; i < network->layerCount; i++) {
        fprintf(fp, "        {\n");
        fprintf(fp, "            \"neuronCount\": %d,\n", network->layers[i].neuronCount);
        fprintf(fp, "            \"neurons\": [\n");
        for (int j = 0; j < network->layers[i].neuronCount; j++) {
            fprintf(fp, "                {\n");
            if (i != 0) {
                fprintf(fp, "                    \"incomingWeights\": [");
                for (int k = 0; k < network->layers[i - 1].neuronCount; k++) {
                    fprintf(fp, "%f", network->layers[i].incomingWeights[j * network->layers[i - 1].neuronCount + k]);
                    if (k == network->layers[i - 1].neuronCount - 1) {
                        fprintf(fp, "],\n");
                    } else {
                        fprintf(fp, ", ");
                    }
                }
            }

            fprintf(fp, "                    \"bias\": %f\n", network->layers[i].biases[j]);

            if (j == network->layers[i].neuronCount - 1) {
                fprintf(fp, "                }\n");
            } else {
                fprintf(fp, "                },\n");
            }
        }
        if (i == network->layerCount - 1) {
            fprintf(fp, "            ]\n");
        } else {
            fprintf(fp, "            ]\n");
        }
        if (i == network->layerCount - 1) {
            fprintf(fp, "        }\n");
        } else {
            fprintf(fp, "        },\n");
        }
    }
    fprintf(fp, "    ]\n");
    fprintf(fp, "}\n");

    // close file
    fclose(fp);
}

// Import network from file (.json file format)
Network *importNetworkJSON(const char *filename) {
    // Open the file
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Read the layer count
    int layerCount;
    if (fscanf(fp, "{\n    \"layerCount\": %d,\n", &layerCount) != 1) {
        perror("Error reading layer count");
        fclose(fp);
        return NULL;
    }

    // Read activation functions
    int *activationFunctions = (int *)malloc((layerCount - 1) * sizeof(int));
    if (activationFunctions == NULL) {
        perror("Memory allocation failed for activation functions");
        fclose(fp);
        return NULL;
    }
    fscanf(fp, "    \"activationFunctions\": [\n");
    for (int i = 0; i < layerCount - 1; i++) {
        if (fscanf(fp, "        %d%*c", &activationFunctions[i]) != 1) {
            perror("Error reading activation functions");
            free(activationFunctions);
            fclose(fp);
            return NULL;
        }
    }
    fscanf(fp, "    ],\n");

    // Read neuron counts and create the network
    int *neuronCounts = (int *)malloc(layerCount * sizeof(int));
    int *hiddenNeuronCounts = (int *)malloc((layerCount-2) * sizeof(int));
    if (neuronCounts == NULL) {
        perror("Memory allocation failed for neuron counts");
        free(activationFunctions);
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "    \"layers\": [\n");
    for (int i = 0; i < layerCount; i++) {
        fscanf(fp, "        {\n            \"neuronCount\": %d,\n", &neuronCounts[i]);
        // set hidden neuron count array
        if (i > 0 && i < layerCount-1) {
            hiddenNeuronCounts[i-1] = neuronCounts[i];
        }
        fscanf(fp, "            \"neurons\": [\n");
        for (int j = 0; j < neuronCounts[i]; j++) {
            fscanf(fp, "%*[^\n]\n"); // skip first curly bracket for neuron

            if (i != 0) { // Skip incomingWeights for now
                fscanf(fp, "%*[^\n]\n");  // Skips the rest of the line (incomingWeights)
            }
            fscanf(fp, "%*[^\n]\n");  // Skips the biases

            fscanf(fp, "%*[^\n]\n"); // skip final curly bracket for neuron
        }
        fscanf(fp, "            ]\n");  // Skip the remaining part of the neurons array
        fscanf(fp, "%*[^\n]\n"); // Skips the final curly bracket for the layer
    }



    // Use createNetwork to allocate the structure
    Network *network = createNetwork(neuronCounts[0], layerCount - 2, hiddenNeuronCounts, neuronCounts[layerCount - 1], activationFunctions);
    if (network == NULL) {
        perror("Failed to create network structure");
        free(activationFunctions);
        free(neuronCounts);
        fclose(fp);
        return NULL;
    }

    // Read weights and biases into the allocated network structure
    rewind(fp);
    fscanf(fp, "{\n    \"layerCount\": %*d,\n    \"activationFunctions\": [\n");
    for (int i = 0; i < layerCount - 1; i++) {
        fscanf(fp, "        %*d%*c");
    }
    fscanf(fp, "    ],\n    \"layers\": [\n");

    for (int i = 0; i < layerCount; i++) {
        fscanf(fp, "        {\n            \"neuronCount\": %*d,\n            \"neurons\": [\n");
        for (int j = 0; j < neuronCounts[i]; j++) {
            fscanf(fp, "                {\n");
            if (i != 0) { // Read incoming weights if not the input layer
                fscanf(fp, "                    \"incomingWeights\": [");
                for (int k = 0; k < neuronCounts[i - 1]; k++) {
                    fscanf(fp, "%f", &network->layers[i].incomingWeights[j * neuronCounts[i - 1] + k]);
                    if (k != neuronCounts[i - 1] - 1) {
                        fscanf(fp, ", ");
                    }
                }
                fscanf(fp, "],\n");
            }

            // Read biases
            fscanf(fp, "                    \"bias\": %f\n", &network->layers[i].biases[j]);
            fscanf(fp, "                }%*[\n,]%*[\n]");
        }
        fscanf(fp, "            ]\n");
        fscanf(fp, "        }%*[\n,]%*[\n]");
    }

    // Clean up and return
    fclose(fp);
    free(neuronCounts);
    free(activationFunctions);
    return network;
}