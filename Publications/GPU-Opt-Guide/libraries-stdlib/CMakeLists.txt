
add_example_with_mkl(rng_test)

add_example(external_rand)
target_compile_options(external_rand PRIVATE -DGPU)
