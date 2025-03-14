#include <math.h>

#include "activationFunctions.h"

// Activation function (sigmoid)
// input is a vector of values
// the result is stored in a vector passed in as a parameter
void sigmoid(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = 1 / (1 + expf(-values[i]));
    }
}

// Derivative of the sigmoid function (uses calculus chain rule)
void sigmoidDerivative(float *values, float *returnValues, int length) {
    float currentSigmoidValue;
    for (int i = 0; i < length; i++) {
        currentSigmoidValue = 1 / (1 + expf(-values[i]));
        returnValues[i] = currentSigmoidValue * (1 - currentSigmoidValue);
    }
}

// Activation function (ReLU)
void relu(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = values[i] > 0 ? values[i] : 0;
    }
}

// Derivative of the ReLU function
void reluDerivative(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = values[i] > 0 ? 1 : 0;
    }
}

// Activation function (Tanh)
void tanhActivation(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        // prevent overflow by cutting off values at 20
        if (values[i] > 20) {
            returnValues[i] = 1;
        } else if (values[i] < -20) {
            returnValues[i] = -1;
        } else {
            returnValues[i] = (expf(values[i]) - expf(-values[i])) / (expf(values[i]) + expf(-values[i]));
        }
    }
}

// Derivative of the tanh function
void tanhDerivative(float *values, float *returnValues, int length) {
    float currentTanhValue;
    for (int i = 0; i < length; i++) {
        currentTanhValue = (expf(values[i]) - expf(-values[i])) / (expf(values[i]) + expf(-values[i]));
        returnValues[i] = 1 - (currentTanhValue * currentTanhValue);
    }
}

// Activation function (Linear)
void linear(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = values[i];
    }
}

// Derivative of the linear function
void linearDerivative(float *values, float *returnValues, int length) {
    // silence unused parameter warning
    (void)values;
    for (int i = 0; i < length; i++) {
        returnValues[i] = 1;
    }
}

// Activation function (Softmax)
// this must take a vecor as input because it is a probability distribution across a vector
// it will also return a vector
void softmax(float *values, float *returnValues, int length) {
    // find max value in the vector
    float max = values[0];
    for (int i = 1; i < length; i++) {
        if (values[i] > max) {
            max = values[i];
        }
    }

    // calculate the softmax function
    float sumExp = 0.0f;
    for (int i = 0; i < length; i++) {
        returnValues[i] = expf(values[i] - max); // subtract max value for numerical stability
        sumExp += returnValues[i];
    }
    for (int i = 0; i < length; i++) {
        returnValues[i] /= sumExp;
    }
}

// Derivative of the softmax function (unneeded because softmax's derivative is the difference between the predicted and expected values)