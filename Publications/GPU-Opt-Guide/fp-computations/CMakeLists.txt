add_compile_options(-fp-model=fast)
add_definitions(-DFP_SIZE=32)
add_example(test_log_omp)
add_example(test_log_sycl)
