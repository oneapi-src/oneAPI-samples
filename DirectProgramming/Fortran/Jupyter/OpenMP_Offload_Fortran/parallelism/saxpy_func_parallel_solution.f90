! Add combined directive here to
!    1. Offload execution to the GPU, use the cause map(tofrom:y)
!       map(to: x) map(from:is_cpu) map(from:num_teams)
!    2. Create multiple master threads use clause num_teams(NUM_BLOCKS)
!    3. Distribute loop iterations to the various master threads.
        
!$omp target teams distribute num_teams(NUM_BLOCKS) map(tofrom: y) map(to:x) map(from:is_cpu, num_teams)
do ib=1,ARRAY_SIZE, NUM_BLOCKS
        if (ib==1) then
                !Test if target is the CPU host or the GPU device
                is_cpu=omp_is_initial_device()
                !Query number of teams created
                num_teams=omp_get_num_teams()
        end if

        do i=ib, ib+NUM_BLOCKS-1
                y(i) = a*x(i) + y(i)
        end do
end do
!$omp end target teams distribute
