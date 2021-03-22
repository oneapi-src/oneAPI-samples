! Solution Using target data
!$omp target data map(tofrom: x)
    !$omp target map(to:y)
    do i=1, ARRAY_SIZE
        x(i) = x(i) + y(i)
    end do
    !$omp end target

    call init (y, ARRAY_SIZE, 2.0)

    !$omp target map(to:y)
    do i=1, ARRAY_SIZE
        x(i) = x(i) + y(i)
    end do
    !$omp end target
!$omp end target data




! Solution Using target enter/exit/update data
!$omp target enter data map(to:x) map(to:y)
!$omp target
do i=1, ARRAY_SIZE
        x(i) = x(i) + y(i)
end do
!$omp end target

call init (y, ARRAY_SIZE, 2.0)
!$omp target update to(y)

!$omp target
do i=1, ARRAY_SIZE
        x(i) = x(i) + y(i)
end do
!$omp end target
!$omp target exit data map(from:x)
