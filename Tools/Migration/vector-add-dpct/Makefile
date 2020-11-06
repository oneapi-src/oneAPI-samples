CXX = nvcc
TARGET = vector_add
SRCS = src/vector_add.cu

# Use predefined implicit rules and add one for *.cu files.
%.o: %.cu
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CXX) $(SRCS) -o $@

run: $(TARGET)
	./$(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
