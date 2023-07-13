all: prog1 prog2 prog3 prog4 prog5 prog6 prog8 prog9 prog10 prog11 prog12 prog14 prog15

prog1: 01_dpex_Intro/lab/simple_njit_cpu.py
	python -Wignore 01_dpex_Intro/lab/simple_njit_cpu.py

prog2: 01_dpex_Intro/lab/simple_context.py
	python -Wignore 01_dpex_Intro/lab/simple_context.py

prog3: 01_dpex_Intro/lab/simple_2d.py
	python -Wignore 01_dpex_Intro/lab/simple_2d.py 
    
prog4: 02_dpctl_Intro/lab/simple_dpctl_queue.py
	python -Wignore 02_dpctl_Intro/lab/simple_dpctl_queue.py
    
prog5: 02_dpctl_Intro/lab/dpctl_queue_2.py
	python -Wignore 02_dpctl_Intro/lab/dpctl_queue_2.py

prog6: 02_dpctl_Intro/lab/simple_dpctl.py 
	python -Wignore 02_dpctl_Intro/lab/simple_dpctl.py

prog8: 03_dpex_Pairwise_Distance/lab/pairwise_distance.py
	python -Wignore 03_dpex_Pairwise_Distance/lab/pairwise_distance.py --steps 1 --size 1024 --repeat 1 --json result_gpu.json

prog9: 03_dpex_Pairwise_Distance/lab/pairwise_distance_gpu.py
	python -Wignore 03_dpex_Pairwise_Distance/lab/pairwise_distance_gpu.py --steps 1 --size 1024 --repeat 1 --json result_gpu.json
prog10: 03_dpex_Pairwise_Distance/lab/pair_wise_kernel.py
	python -Wignore 03_dpex_Pairwise_Distance/lab/pair_wise_kernel.py --steps 1 --size 1024 --repeat 1 --json result_gpu.json

prog11: 04_dpex_Black_Sholes/lab/black_sholes_jit_cpu.py
	python -Wignore 04_dpex_Black_Sholes/lab/black_sholes_jit_cpu.py --steps 1 --size 1024 --repeat 1 --json result_gpu.json

prog12: 04_dpex_Black_Sholes/lab/black_sholes_jit_gpu.py
	python -Wignore 04_dpex_Black_Sholes/lab/black_sholes_jit_gpu.py --steps 1 --size 1024 --repeat 1 --json result_gpu.json

prog14: 05_dpex_Kmeans/lab/kmeans.py
	python -Wignore 05_dpex_Kmeans/lab/kmeans.py --steps 1 --size 1024 --repeat 1 --json result_gpu.json

prog15: 05_dpex_Kmeans/lab/kmeans_kernel_atomic.py
	python -Wignore 05_dpex_Kmeans/lab/kmeans_kernel_atomic.py --steps 1 --size 1024 --repeat 1 --json result_gpu.json