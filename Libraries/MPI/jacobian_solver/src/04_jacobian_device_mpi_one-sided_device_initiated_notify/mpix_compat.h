/*==============================================================
 * Copyright © 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */
#ifndef MPIX_COMPAT_H
#define MPIX_COMPAT_H

#define MPI_ERR_INVALID_NOTOFOCATION MPI_ERR_OTHER

/* int MPI_Win_notify_attach(MPI_Win win, int notification_num, MPI_Info info); */
#define MPI_Win_notify_attach(win, notification_num, info) \
            MPIX_Win_create_notify(win, notification_num)

/* int MPI_Win_notify_detach(MPI_Win win); */
#define MPI_Win_notify_detach(win) \
            MPIX_Win_free_notify(win)

/* int MPI_Win_notify_get_value(MPI_Win win, int notification_idx, MPI_Count *value) */
#define MPI_Win_notify_get_value(win, notification_idx, value) \
            MPIX_Win_get_notify(win, notification_idx, value)

/* int MPI_Win_notify_set_value(MPI_Win win, int notification_idx, MPI_Count value) */
#define MPI_Win_notify_set_value(win, notification_idx, value) \
            MPIX_Win_set_notify(win, notification_idx, value)

#define MPI_Put_notify MPIX_Put_notify
#define MPI_Get_notify MPIX_Get_notify

#endif /* MPIX_COMPAT_H */
