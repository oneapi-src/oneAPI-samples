all:
	make -C src/01_jacobian_host_mpi_one-sided
	make -C src/02_jacobian_device_mpi_one-sided_gpu_aware
	make -C src/03_jacobian_device_mpi_one-sided_device_initiated

debug:
	make debug -C src/01_jacobian_host_mpi_one-sided
	make debug -C src/02_jacobian_device_mpi_one-sided_gpu_aware
	make debug -C src/03_jacobian_device_mpi_one-sided_device_initiated

clean:
	make clean -C src/01_jacobian_host_mpi_one-sided
	make clean -C src/02_jacobian_device_mpi_one-sided_gpu_aware
	make clean -C src/03_jacobian_device_mpi_one-sided_device_initiated
