#include <stdio.h>
#include <Accelerate/Accelerate.h>  // Include Accelerate framework instead of cblas.h

#include "linearAlgebra.h"

// Define the global variable and set it to BASIC by default
int LIN_ALG_METHOD_SELECTION = BASIC;  // Default to BASIC, can be changed to BLAS as needed

// functions to set and get LIN_ALG_METHOD_SELECTION
void setLinAlgSelectedMethod(int value) {
    LIN_ALG_METHOD_SELECTION = value;
}

int getLinAlgSelectedMethod() {
    return LIN_ALG_METHOD_SELECTION;
}

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
    if (LIN_ALG_METHOD_SELECTION == BLAS) {
        // Use BLAS for matrix multiplication (cblas_sgemm)
        // Note: In row-major order, lda should be the number of columns of matrix A (colsA),
        // as it's the stride between consecutive rows in memory.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rowsA, colsB, colsA, 1.0f, matA, colsA, matB, colsB, 0.0f, matC, colsB);
    } else {
        // default to BASIC
        ///////////////////

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

// Function to add two vectors
void floatVectorAdd(float *vecA, float *vecB, float *vecC, int vecSize) {
    if (LIN_ALG_METHOD_SELECTION == BLAS) {
        // Use BLAS for vector addition (cblas_saxpy)
        cblas_saxpy(vecSize, 1.0f, vecA, 1, vecC, 1);
        cblas_saxpy(vecSize, 1.0f, vecB, 1, vecC, 1);
    } else {
        // default to BASIC
        ///////////////////

        for (int i = 0; i < vecSize; i++) {
            vecC[i] = vecA[i] + vecB[i];
        }
    }
}