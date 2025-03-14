#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include "networkErrors.h"

// multiplies two float matricies
void floatMatrixMultiply(float *matA, float *matB, float *matC, int rowsA, int colsA, int rowsB, int colsB);

// prints a matrix
void printMatrix(float *matrix, int rows, int cols);

#endif