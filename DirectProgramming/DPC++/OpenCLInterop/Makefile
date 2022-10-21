all: prog 

prog: src/sycl_with_opencl_objects.dp.cpp
	dpcpp -lOpenCL -o bin/prog src/sycl_with_opencl_objects.dp.cpp

Run:
	bin/prog	

clean:
	rm -rf bin/prog
