!$omp target map(tofrom: y) map(to:x) map(from:is_cpu)
is_cpu=omp_is_initial_device();
do i=1,ARRAY_SIZE
        y(i) = a*x(i) + y(i)
end do
!$omp end target
