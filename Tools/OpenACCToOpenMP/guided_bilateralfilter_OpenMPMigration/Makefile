# Define the compiler
CC = icx

# Define the compiler flags
CFLAGS = -fiopenmp -fopenmp-targets=spir64

# Define the include directory
#INCLUDES = -I ../../include/

# Define the source files
SRCS = ./openMP_migrated_output/bilateralFilter_translated.c

# Define the object files
OBJS = $(SRCS:.c=.o)

# Define the target executable
TARGET = bilateralFilter

# Default rule to build the target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

# Rule to compile the source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to run the target executable
run: $(TARGET)
	./$(TARGET)

# Clean rule to remove the generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
