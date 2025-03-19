#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

// Define activation code constants
// ACTIVATION CODES: 0 = sigmoid, 1 = relu, 2 = tanh, 3 = linear, 4 = softmax (only for output activation)
#define SIGMOID_ACTIVATION 0
#define RELU_ACTIVATION 1
#define TANH_ACTIVATION 2
#define LINEAR_ACTIVATION 3
#define SOFTMAX_ACTIVATION 4

// Redeclare functions
// These activation functions are used in the forward pass and backwards pass of the neural network. They are used after the weighted sum is calculated to apply a linear transformation on the vector.
// All activation functions take the following input parameters:
// values: an array of numbers. Usually the weighted sum of a layer
// returnValues: the pointer to the returning array. Usually the values of a network layer
// length: the size of the two value arrays
void sigmoid(float *values, float *returnValues, int length);
void sigmoidDerivative(float *values, float *returnValues, int length);
void relu(float *values, float *returnValues, int length);
void reluDerivative(float *values, float *returnValues, int length);
void tanhActivation(float *values, float *returnValues, int length);
void tanhDerivative(float *values, float *returnValues, int length);
void linear(float *values, float *returnValues, int length);
void linearDerivative(float *values, float *returnValues, int length);
void softmax(float *values, float *returnValues, int length);

// define ActivationFunction type and declare activation function & derivative arrays
typedef void (*ActivationFunction)(float*, float*, int);
extern ActivationFunction activationFunctionsPtr[];
extern ActivationFunction activationFunctionDerivativesPtr[];

#endif