all:
	mkdir bin

	icpx -fsycl 01_SYCL_Migration_Simple_VectorAdd/sycl_migrated/*.cpp -I 01_SYCL_Migration_Simple_VectorAdd/sycl_migrated/include -o bin/app1 

	icpx -fsycl 01_SYCL_Migration_Simple_VectorAdd/sycl_migrated_optimized/*.cpp -I 01_SYCL_Migration_Simple_VectorAdd/sycl_migrated/include -o bin/app2

	icpx -fsycl -I 02_SYCL_Migration_SortingNetworks/sycl_migrated/Common -I 02_SYCL_Migration_SortingNetworks/sycl_migrated/include 02_SYCL_Migration_SortingNetworks/sycl_migrated/Samples/2_Concepts_and_Techniques/sortingNetworks/*.cpp -o bin/app3 -w

	icpx -fsycl -I 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated/Common -I 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated/include 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated/Samples/3_CUDA_Features/jacobiCudaGraphs/*.cpp -o bin/app4 -w

	icpx -fsycl -I 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated_optimized/Common -I 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated_optimized/include 03_SYCL_Migration_Jacobi_Iterative/sycl_migrated_optimized/Samples/3_CUDA_Features/jacobiCudaGraphs/*.cpp -o bin/app5 -w

	icpx -fsycl -I 04_SYCL_Migration_MatrixMul_CuBlas/sycl_migrated/Common -I 04_SYCL_Migration_MatrixMul_CuBlas/sycl_migrated/include 04_SYCL_Migration_MatrixMul_CuBlas/sycl_migrated/Samples/4_CUDA_Libraries/matrixMulCUBLAS/*.cpp -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -o bin/app6 -w

	icpx -fsycl -fsycl-targets=intel_gpu_pvc -I 05_SYCL_Migration_OceanFFT/sycl_migrated/Common -I 05_SYCL_Migration_OCeanFFT/sycl_migrated/data -I 05_SYCL_Migration_OceanFFT/sycl_migrated/include 05_SYCL_Migration_OceanFFT/sycl_migrated/Samples/4_CUDA_Libraries/oceanFFT/*.cpp -qmkl -o bin/app7 -w && cp -rf 05_SYCL_Migration_OceanFFT/sycl_migrated/data/* bin/

	icpx -fsycl -fsycl-targets=intel_gpu_pvc -I 06_SYCL_Migration_SimpleCUDAGraphs/sycl_migrated_option1/Common -I 06_SYCL_Migration_SimpleCUDAGraphs/sycl_migrated_option1/include 06_SYCL_Migration_SimpleCUDAGraphs/sycl_migrated_option1/Samples/3_CUDA_Features/simpleCudaGraphs/*.cpp -pthread -o bin/app8 -w

	icpx -fsycl -fsycl-targets=intel_gpu_pvc -I 06_SYCL_Migration_SimpleCUDAGraphs/sycl_migrated_option2/Common -I 06_SYCL_Migration_SimpleCUDAGraphs/sycl_migrated_option2/include 06_SYCL_Migration_SimpleCUDAGraphs/sycl_migrated_option2/Samples/3_CUDA_Features/simpleCudaGraphs/*.cpp -pthread -o bin/app9 -w

	icpx -fsycl -fsycl-targets=intel_gpu_pvc -I 07_SYCL_Migration_MonteCarloMultiGPU/sycl_migrated/Common -I 07_SYCL_Migration_MonteCarloMultiGPU/sycl_migrated/include 07_SYCL_Migration_MonteCarloMultiGPU/sycl_migrated/Samples/5_Domain_Specific/MonteCarloMultiGPU/*.cpp -pthread -qmkl -o bin/app10 -w
	
run:
	bin/app1
	bin/app2
	bin/app3
	bin/app4
	bin/app5
	bin/app6
	bin/app7 qatest
	bin/app8
	bin/app9
	bin/app10

clean:
	rm -rf bin

