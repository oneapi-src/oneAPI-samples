all:
	make -C src/01_jacobian_host_mpi_one-sided
	make -C src/02_jacobian_device_mpi_one-sided_gpu_aware
	make -C src/03_jacobian_device_mpi_one-sided_device_initiated
	make -C src/04_jacobian_device_mpi_one-sided_device_initiated_notify

debug:
	make debug -C src/01_jacobian_host_mpi_one-sided
	make debug -C src/02_jacobian_device_mpi_one-sided_gpu_aware
	make debug -C src/03_jacobian_device_mpi_one-sided_device_initiated
	make debug -C src/04_jacobian_device_mpi_one-sided_device_initiated_notify

clean:
	make clean -C src/01_jacobian_host_mpi_one-sided
	make clean -C src/02_jacobian_device_mpi_one-sided_gpu_aware
	make clean -C src/03_jacobian_device_mpi_one-sided_device_initiated
	make clean -C src/04_jacobian_device_mpi_one-sided_device_initiated_notify
