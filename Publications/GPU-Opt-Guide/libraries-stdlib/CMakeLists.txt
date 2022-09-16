
add_example_with_mkl(rng_test)
target_compile_options(rng_test PRIVATE -DHOST)

add_example(external_rand)
target_compile_options(external_rand PRIVATE -DHOST)
