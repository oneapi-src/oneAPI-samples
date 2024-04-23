        !=============================================================
        ! Copyright © 2022 Intel Corporation
        !
        ! SPDX-License-Identifier: MIT
        !=============================================================
MODULE test

    USE ISO_C_BINDING

CONTAINS

  SUBROUTINE log_real_sp (nelements, nrepetitions, initial_value, res) bind(C,NAME='log_real_sp')
    IMPLICIT NONE
    INTEGER(KIND=C_INT), VALUE :: nelements, nrepetitions
    REAL(C_FLOAT), VALUE :: initial_value
    REAL(C_FLOAT) :: res(0:nelements-1), tmp
    INTEGER :: i, j

    !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO PRIVATE(tmp)
    DO j = 0, nelements-1
      tmp = initial_value
      DO i = 0, nrepetitions-1
        tmp = tmp + log(tmp)
      END DO
      res(j) = tmp
    END DO
    RETURN
  END SUBROUTINE log_real_sp

  SUBROUTINE log_real_dp (nelements, nrepetitions, initial_value, res) bind(C,NAME='log_real_dp')
    IMPLICIT NONE
    INTEGER(KIND=C_INT), VALUE :: nelements, nrepetitions
    REAL(C_DOUBLE), VALUE :: initial_value
    REAL(C_DOUBLE) :: res(0:nelements-1), tmp
    INTEGER :: i, j

    !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO PRIVATE(tmp)
    DO j = 0, nelements-1
      tmp = initial_value
      DO i = 0, nrepetitions-1
        tmp = tmp + log(tmp)
      END DO
      res(j) = tmp
    END DO
    RETURN
  END SUBROUTINE log_real_dp

END MODULE test
