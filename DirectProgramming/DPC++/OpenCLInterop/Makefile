all: prog1 prog2 

prog1: src/dpcpp_with_opencl_kernel.dp.cpp
	dpcpp -o bin/prog1 src/dpcpp_with_opencl_kernel.dp.cpp

prog2: src/dpcpp_with_opencl_objects.dp.cpp
	dpcpp -lOpenCL -o bin/prog2 src/dpcpp_with_opencl_objects.dp.cpp

#Run: prog1 prog2

Run:
	bin/prog1
	bin/prog2	

clean:
	rm -rf bin/prog1 bin/prog2
