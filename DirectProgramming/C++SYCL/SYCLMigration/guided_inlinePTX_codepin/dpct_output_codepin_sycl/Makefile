##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=icpx
CFLAGS=-fsycl
CC_LIBS=


# Include header file diretory:
INC_DIR=-I ../dpct_output/include/
COMMON_DIR=-I Common

TARGET= inlinePTX_sycl
OBJS= Samples/2_Concepts_and_Techniques/inlinePTX/inlinePTX.dp.cpp

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(INC_DIR) $(COMMON_DIR) -o $(TARGET)


run: $(TARGET)
	ONEAPI_DEVICE_SELECTOR="opencl:cpu" ./$(TARGET)

clean:
	rm -rf $(TARGET) 
