# Variables
CC = clang
CFLAGS = -Wall -Wextra -O2 -Iinclude -I/opt/homebrew/opt/openblas/include  # Add flags for new LAPACK interface
CFLAGS += -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64  # Enable new CBLAS interface and ILP64 support
LDFLAGS = -framework Accelerate  # Link against Accelerate framework

SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

# Files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
LIB = $(LIB_DIR)/libneuralNetworkLib.a

# Rules
all: $(LIB)

# Create static library
$(LIB): $(OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)

# Install header files (optional)
install:
	@mkdir -p /usr/local/include/neuralNetworkLib
	cp -r include/* /usr/local/include/neuralNetworkLib
	@mkdir -p /usr/local/lib
	cp $(LIB) /usr/local/lib

# Uninstall library (optional)
uninstall:
	rm -rf /usr/local/include/neuralNetworkLib
	rm -f /usr/local/lib/libneuralNetworkLib.a