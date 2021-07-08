all: prog1 prog2 prog3

prog1: intro/simple_solution.cpp
	icpx -g -fiopenmp -fopenmp-targets=spir64 -o intro/bin/prog1 intro/simple_solution.cpp

prog2: datatransfer/main_test.cpp
	icpx -g -fiopenmp -fopenmp-targets=spir64 -o datatransfer/bin/prog2 datatransfer/main_test.cpp

prog3: parallelism/main_test.cpp
	icpx -g -fiopenmp -fopenmp-targets=spir64 -o parallelism/bin/prog3 parallelism/main_test.cpp

#Run: prog1 prog2 prog3

Run:
	intro/bin/prog1
	datatransfer/bin/prog2
	parallelism/bin/prog3

clean:
	rm -rf intro/bin/prog1 datatransfer/bin/prog2 parallelism/bin/prog3
