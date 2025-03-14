#include <stdio.h>

#include "linearAlgebra.h"

// Function to multiply two float matrices
// the result will be a matrix with dimensions rowsA x colsB
// Example:
//
//    A (2x3)                B (3x2)                               C (2x2)                               C (2x2)
// [ 1   2   3 ]     x     [  7   8  ]     =     [ (1*7 + 2*9 + 3*11)   (1*8 + 2*10 + 3*12) ]     =   [  58   64  ]
// [ 4   5   6 ]           [  9  10  ]           [ (4*7 + 5*9 + 6*11)   (4*8 + 5*10 + 6*12) ]         [ 139  154  ]
//                         [ 11  12  ]
//
void floatMatrixMultiply(float *matA, float *matB, float *matC, int rowsA, int colsA, int rowsB, int colsB) {
    // Ensure that the dimensions match for multiplication
    if (colsA != rowsB) {
        handleError("floatMatrixMultiply dim mismatch. colsA must equal rowsB");
        return;
    }

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            // initialize resulting matrix to zero
            matC[i * colsB + j] = 0; // matC[i][j]
            
            // do the matrix multiplication
            for (int k = 0; k < colsA; k++) {
                matC[i * colsB + j] += matA[i * colsA + k] * matB[k * colsB + j];
            }
        }
    }
}

// Helper to print matricies
void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        printf("[ ");
        for (int j = 0; j < cols; j++) {
            if (j == cols-1) {
                printf("%f", matrix[i * cols + j]);
            } else {
                printf("%f, ", matrix[i * cols + j]);
            }
        }
        printf(" ]\n");
    }
}