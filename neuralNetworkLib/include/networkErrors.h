#ifndef NETWORKERRORS_H
#define NETWORKERRORS_H

#define SUCCESS 0
#define ERR_GENERAL -1
#define ERR_ALLOC -2
#define ERR_DIM_MISMATCH -3
#define ERR_UNEXPECTED_VAL -4

// prints an error message. basically useless since this is just a wrapper for fprintf that does nothing more
void handleError(const char * message);

#endif