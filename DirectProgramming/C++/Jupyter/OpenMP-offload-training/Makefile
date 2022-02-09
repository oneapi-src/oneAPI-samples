all: prog1 prog2 prog3 prog4

prog1: intro/simple_solution.cpp
	icpx -g -fiopenmp -fopenmp-targets=spir64 -o intro/bin/prog1 intro/simple_solution.cpp

prog2: datatransfer/main_test.cpp
	icpx -g -fiopenmp -fopenmp-targets=spir64 -o datatransfer/bin/prog2 datatransfer/main_test.cpp

prog3: parallelism/main_test.cpp
	icpx -g -fiopenmp -fopenmp-targets=spir64 -o parallelism/bin/prog3 parallelism/main_test.cpp
    
prog4: USM/usm_explicit.cpp
	icpx -g -fiopenmp -fopenmp-targets=spir64 -o USM/bin/prog4 USM/usm_explicit.cpp

#Run: prog1 prog2 prog3 prog4

Run:
	intro/bin/prog1
	datatransfer/bin/prog2
	parallelism/bin/prog3
	USM/bin/prog4

clean:
	rm -rf intro/bin/prog1 datatransfer/bin/prog2 parallelism/bin/prog3 USM/bin/prog4
