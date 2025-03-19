#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include "networkErrors.h"

// Global variable to choose between BASIC, BLAS, or Accelerate methods
// The user should set this variable carefully before running matrix/vector operations
//   - BASIC: Uses custom implementations of matrix multiplication and vector addition
//   - BLAS: Uses cross-platform BLAS library functions for better performance (under the accelerate library on Apple silicon and optimized for the platform)

enum Method { BASIC, BLAS };

// Declare the global variable as extern here
extern int LIN_ALG_METHOD_SELECTION;  // Declaration only

// functions to set and get LIN_ALG_METHOD_SELECTION
void setLinAlgSelectedMethod(int value);
int getLinAlgSelectedMethod();

// multiplies two float matricies
void floatMatrixMultiply(float *matA, float *matB, float *matC, int rowsA, int colsA, int rowsB, int colsB);

// prints a matrix
void printMatrix(float *matrix, int rows, int cols);

// adds two vectors
void floatVectorAdd(float *vecA, float *vecB, float *vecC, int vecSize);

#endif