#include <stdio.h>

#include "networkErrors.h"

void handleError(const char * message) {
    fprintf(stderr, "Error: %s\n", message);
}