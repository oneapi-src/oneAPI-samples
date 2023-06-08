# to compile and run the sequential version type: make seq
# to compile and run the coarray version type: make coarray

FC=ifx
FCFLAGS=
FCFLAGS_COARRAY=-coarray

SEQ=./seq 
SEQ_OBJS= mcpi_sequential.o
COARRAY=./coarray
COARRAY_OBJS= mcpi_coarray_final.o

mcpi_sequential.o: ./src/mcpi_sequential.F90
mcpi_coarray.o: ./src/mcpi_coarray_final.F90

$(SEQ): $(SEQ_OBJS)
	$(FC) $(SEQ_OBJS) -o $@ 
	$(SEQ)

$(COARRAY): $(COARRAY_OBJS)
	$(FC) $(FCFLAGS_COARRAY) $(COARRAY_OBJS) -o $@ 
	$(COARRAY)

all: $(SEQ) $(COARRAY)

mcpi_sequential.o: ./src/mcpi_sequential.F90
	$(FC) $(FCFLAGS) -c $< -o $@

mcpi_coarray_final.o: ./src/mcpi_coarray_final.F90
	$(FC) $(FCFLAGS_COARRAY) -c $< -o $@

.DEFAULT_GOAL=$(SEQ)

.PHONY: clean 

clean:
	-/bin/rm -f *.o $(SEQ) $(COARRAY)
