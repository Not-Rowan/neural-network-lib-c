#include "errorFunctions.h"

// Cost function (squared error)
// just like activation functions, this must take a vector as input and return a vector
void squaredError(float *expected, float *actual, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = (expected[i] - actual[i]) * (expected[i] - actual[i]);
    }
}

// Derivative of the squared error cost function
void squaredErrorDerivative(float *expected, float *actual, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = 2 * (actual[i] - expected[i]);
    }
}