all: prog1 prog2 prog3

prog1: intro/simple_solution.f90
	ifx -debug offload -fiopenmp -fopenmp-targets=spir64 -o intro/bin/prog1 intro/simple_solution.f90

prog2: datatransfer/main_test.f90
	ifx -debug offload -fiopenmp -fopenmp-targets=spir64 -o datatransfer/bin/prog2 datatransfer/main_test.f90

prog3: parallelism/main_test.f90
	ifx -debug offload -fiopenmp -fopenmp-targets=spir64 -o parallelism/bin/prog3 parallelism/main_test.f90

#Run: prog1 prog2 prog3

Run:
	intro/bin/prog1
	datatransfer/bin/prog2
	parallelism/bin/prog3

clean:
	rm -rf intro/bin/prog1 datatransfer/bin/prog2 parallelism/bin/prog3
