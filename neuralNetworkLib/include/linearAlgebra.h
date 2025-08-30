#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include "networkErrors.h"

// Global variable to choose between BASIC, BLAS, or Accelerate methods
// The user should set this variable carefully before running matrix/vector operations
//   - BASIC (0): Uses custom implementations of matrix multiplication and vector addition
//   - BLAS (1): Uses cross-platform BLAS library functions for better performance (under the accelerate library on Apple silicon and optimized for the platform)
enum Method { BASIC, BLAS };

// Declare the global variable as extern here
extern int LIN_ALG_METHOD_SELECTION;  // Declaration only


// setLinAlgSelectedMethod() sets the LIN_ALG_METHOD_SELECTION variable
// Parameters:
//     - value: BASIC or BLAS from Method enum
// Return:
//     - Nothing
void setLinAlgSelectedMethod(int value);

// getLinAlgSelectedMethod() returns the selected linear algebra method (BASIC or BLAS)
// Parameters:
//     - None
// Return:
//     - Enum code (0 for BASIC and 1 for BLAS)
int getLinAlgSelectedMethod();

// floatMatrixMultiply() multiplies two float matrices into a return matrix (matC)
// Parameters:
//     - matA: input matrix A
//     - matB: input matrix B
//     - matC: output matrix C
//     - rowsA: num of rows matrix A has
//     - colsA: num of cols matrix A has
//     - rowsB: num of rows matrix B has
//     - colsB: num of cols matrix B has
// Return:
//     - Nothing
void floatMatrixMultiply(float *matA, float *matB, float *matC, int rowsA, int colsA, int rowsB, int colsB);

// printMatrix() prints the structure and values of a given matrix
// Parameters:
//     - matrix: matrix to print
//     - rows: number of rows that matrix has
//     - cols: number of cols that matrix has
// Return:
//     - None
void printMatrix(float *matrix, int rows, int cols);

// floatVectorAdd() adds two float vectors into a return vector (vecC)
// Parameters:
//     - vecA: input vector A
//     - vecB: input vector B
//     - vecC: output vector C
//     - vecSize: the size of the vector
// Return:
//     - None
void floatVectorAdd(float *vecA, float *vecB, float *vecC, int vecSize);

#endif
