all: prog 

prog: src/dpcpp_with_opencl_objects.dp.cpp
	dpcpp -lOpenCL -o bin/prog src/dpcpp_with_opencl_objects.dp.cpp

Run:
	bin/prog	

clean:
	rm -rf bin/prog
