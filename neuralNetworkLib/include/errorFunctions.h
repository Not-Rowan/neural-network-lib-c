#ifndef ERRORFUNCTIONS_H
#define ERRORFUNCTIONS_H

// Redeclare functions

// These error functions are used to calculate the error between a real and expected value
// They take the following input parameters
// expected: the expected values as an array
// actual: the actual values that the network has produced
// returnValues: the return array to place the error value in for each index
// length: the length of the input arrays
void squaredError(float *expected, float *actual, float *returnValues, int length);
void squaredErrorDerivative(float *expected, float *actual, float *returnValues, int length);

#endif