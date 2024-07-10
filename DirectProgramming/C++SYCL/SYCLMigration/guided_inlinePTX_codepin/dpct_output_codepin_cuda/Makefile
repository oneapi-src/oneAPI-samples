##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=nvcc
CFLAGS=
CC_LIBS=


# Include header file diretory:
INC_DIR=-I ../dpct_output/include/
COMMON_DIR=-I Common
TARGET= inlinePTX_cuda
OBJS= Samples/2_Concepts_and_Techniques/inlinePTX/inlinePTX.cu


all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(INC_DIR) $(COMMON_DIR) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(TARGET) 
