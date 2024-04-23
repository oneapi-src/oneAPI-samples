all:
	mkdir bin

	icpx -fsycl 01_SYCL_Migration_Simple_VectorAdd/sycl_migrated/*.cpp -I 01_SYCL_Migration_Simple_VectorAdd/sycl_migrated/include -o bin/app1 

	icpx -fsycl 01_SYCL_Migration_Simple_VectorAdd/sycl_migrated_optimized/*.cpp -I 01_SYCL_Migration_Simple_VectorAdd/sycl_migrated/include -o bin/app2

	icpx -fsycl -I 02_SYCL_Migration_SortingNetworks/sycl_migrated/Common -I 02_SYCL_Migration_SortingNetworks/sycl_migrated/include 02_SYCL_Migration_SortingNetworks/sycl_migrated/Samples/2_Concepts_and_Techniques/sortingNetworks/*.cpp -o bin/app3 -w

	icpx -fsycl -I 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated/Common -I 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated/include 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated/Samples/3_CUDA_Features/jacobiCudaGraphs/*.cpp -o bin/app4 -w

	icpx -fsycl -I 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated_optimized/Common -I 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated_optimized/include 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated_optimized/Samples/3_CUDA_Features/jacobiCudaGraphs/*.cpp -o bin/app5 -w

	icpx -fsycl -I 04_SYCL_Migration_MatrixMul_CuBlas/sycl_migrated/Common -I 04_SYCL_Migration_MatrixMul_CuBlas/sycl_migrated/include 04_SYCL_Migration_MatrixMul_CuBlas/sycl_migrated/Samples/4_CUDA_Libraries/matrixMulCUBLAS/*.cpp -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -o bin/app6 -w

run:
	bin/app1
	bin/app2
	bin/app3
	bin/app4
	bin/app5
	bin/app6

clean:
	rm -rf bin

