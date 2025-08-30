#ifndef NETWORKUTILITIES_H
#define NETWORKUTILITIES_H

#include "networkStructure.h"


// printNetwork() prints the structure of the network separated into layers with node values, weights, and biases
// Parameters:
//     - network: pointer to the network
// Return:
//     - Nothing
void printNetwork(Network *network);

// copyNetwork() copies one network structure to another
// Assumes that the destination network has the same structure as the source network and that the destination network has already been allocated using createNetwork()
// Parameters:
//     - destination: pointer to the destination network
//     - source: pointer to the source network
// Return:
//     - Nothing
void copyNetwork(Network *destination, Network *source);

// exportNetworkJSON() exports the network and it's parameters into JSON format
// Parameters:
//     - network: pointer to the network
//     - filename: the name of the file to save the network to (as a string, e.g. "network1.json")
// Return:
//     - Nothing
void exportNetworkJSON(Network *network, char *filename);

// importNetworkJSON() imports the network from a JSON file
// Parameters:
//     - filename: the name of the file containing the network JSON data (as a string, e.g. "network1.json")
// Return:
//     - a pointer to the network
Network *importNetworkJSON(const char *filename);

#endif