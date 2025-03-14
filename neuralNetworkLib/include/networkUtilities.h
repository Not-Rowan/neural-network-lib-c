#ifndef NETWORKUTILITIES_H
#define NETWORKUTILITIES_H

#include "networkStructure.h"

// Redeclare functions
void printNetwork(Network *network);
void copyNetwork(Network *destination, Network *source);
void exportNetworkJSON(Network *network, char *filename);
Network *importNetworkJSON(const char *filename);

#endif