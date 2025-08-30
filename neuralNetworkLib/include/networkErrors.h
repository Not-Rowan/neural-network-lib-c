#ifndef NETWORKERRORS_H
#define NETWORKERRORS_H

#define SUCCESS 0
#define ERR_GENERAL -1
#define ERR_ALLOC -2
#define ERR_DIM_MISMATCH -3
#define ERR_UNEXPECTED_VAL -4

// handleError() prints an error to stderr using fprintf based on a given message
// Parameters:
//     - message: a string containing the error message (e.g. "unexpected value for function1().")
// Return:
//     - Nothing
void handleError(const char * message);

#endif