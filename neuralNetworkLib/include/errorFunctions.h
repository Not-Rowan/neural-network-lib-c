#ifndef ERRORFUNCTIONS_H
#define ERRORFUNCTIONS_H

// Redeclare functions

// These error functions are used to calculate the error between a real and expected value
// They take the following input parameters
// expected: the expected values as an array
// actual: the actual values that the network has produced
// returnValues: the return array to place the error value in for each index
// length: the length of the input arrays

// squaredError() takes the difference between the expected vector and actual vector and squares it, then placing it into a return vector (returnValues)
// Parameters:
//     - expected: vector of expected values
//     - actual: vector of actual values
//     - returnValues: return vector containing the squared error of the two inputs
//     - length: the length of the input vectors 
// Return:
//     - None
void squaredError(float *expected, float *actual, float *returnValues, int length);

// squaredErrorDerivative() is the derivative version of squaredError()
// Parameters:
//     - expected: vector of expected values
//     - actual: vector of actual values
//     - returnValues: return vector containing the squared error of the two inputs
//     - length: the length of the input vectors 
// Return:
//     - None
void squaredErrorDerivative(float *expected, float *actual, float *returnValues, int length);

#endif