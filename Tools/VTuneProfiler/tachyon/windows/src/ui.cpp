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

/*
 * ui.cpp - Contains functions for dealing with user interfaces
 */

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "util.h"
#include "ui.h"

static void (* rt_static_ui_message) (int, const char *) = NULL;
static void (* rt_static_ui_progress) (int) = NULL;
static int (* rt_static_ui_checkaction) (void) = NULL;

extern bool silent_mode;

void set_rt_ui_message(void (* func) (int, const char *)) {
  rt_static_ui_message = func;
}

void set_rt_ui_progress(void (* func) (int)) {
  rt_static_ui_progress = func;
}

void rt_ui_message(int level, const char * msg) {
  if (rt_static_ui_message == NULL) {
    if ( !silent_mode ) {
      fprintf(stderr, "%s\n", msg);
      fflush (stderr);
    }
  } else {
    rt_static_ui_message(level, msg);
  }
}

void rt_ui_progress(int percent) {
  if (rt_static_ui_progress != NULL)
    rt_static_ui_progress(percent);
  else {
    if ( !silent_mode ) {
      fprintf(stderr, "\r %3d%% Complete            \r", percent);
      fflush(stderr);
    }
  }
}

int rt_ui_checkaction(void) {
  if (rt_static_ui_checkaction != NULL) 
    return rt_static_ui_checkaction();
  else
    return 0;
}














