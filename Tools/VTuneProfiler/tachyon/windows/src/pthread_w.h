//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================


/*
    The original source for this example is
    Copyright (c) 1994-2008 John E. Stone
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. The name of the author may not be used to endorse or promote products
       derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.
*/

#ifdef EMULATE_PTHREADS

#ifndef _PTHREAD_H_DEFINED
#define _PTHREAD_H_DEFINED

#include <windows.h>
#include <errno.h>
#ifndef ENOTSUP
#define ENOTSUP EPERM
#endif

/*  just need <stddef.h> on Windows to get size_t defined  */
#include <stddef.h>

#define ERROR_PTHREAD 1000
#define ERROR_MODE 1001
#define ERROR_UNIMPL 1002

/*
    Basics
*/

struct pthread_s {
    HANDLE winthread_handle;
    DWORD winthread_id;
};
typedef struct pthread_s *pthread_t;  /*  one of the few types that's pointer, not struct  */

typedef struct {
    int i;  /*  not yet defined...  */
} pthread_attr_t;

/*
    Mutex
*/

typedef struct {
    int i;  /*  not yet defined...  */
} pthread_mutexattr_t;

typedef struct {
    CRITICAL_SECTION critsec;
} pthread_mutex_t;

/*
    Function prototypes
*/

extern int pthread_create (pthread_t *thread, pthread_attr_t *attr, void *(*start_routine) (void *), void *arg);
extern int pthread_join (pthread_t th, void **thread_return);
extern void pthread_exit (void *retval);

extern int pthread_mutex_init (pthread_mutex_t *mutex, pthread_mutexattr_t *mutex_attr);
extern int pthread_mutex_destroy (pthread_mutex_t *mutex);
extern int pthread_mutex_lock (pthread_mutex_t *mutex);
extern int pthread_mutex_unlock (pthread_mutex_t *mutex);

#endif  /*  _PTHREAD_H_DEFINED  */

#endif  /*  EMULATE_PTHREADS  */
