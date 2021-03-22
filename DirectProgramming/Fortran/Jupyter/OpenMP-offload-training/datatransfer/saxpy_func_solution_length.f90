!$omp target map(tofrom: y(1:ARRAY_SIZE)) map(to:x(1:ARRAY_SIZE)) map(from:is_cpu)
is_cpu=omp_is_initial_device();
do i=1,ARRAY_SIZE
        y(i) = a*x(i) + y(i)
end do
!$omp end target
