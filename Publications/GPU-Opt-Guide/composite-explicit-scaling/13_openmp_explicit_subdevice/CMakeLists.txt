
add_example(openmp_explicit_subdevice)
target_compile_options(openmp_explicit_subdevice
  PUBLIC -O3 -fiopenmp -fopenmp-targets=spir64
  )
target_link_options(openmp_explicit_subdevice
  PUBLIC -fiopenmp -fopenmp-targets=spir64
)
