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
 * plane.cpp - This file contains the functions for dealing with planes.
 */
 
#include "machine.h"
#include "types.h"
#include "macros.h"
#include "vector.h"
#include "intersect.h"
#include "util.h"

#define PLANE_PRIVATE
#include "plane.h"

static object_methods plane_methods = {
  (void (*)(void *, void *))(plane_intersect),
  (void (*)(void *, void *, void *, void *))(plane_normal),
  plane_bbox, 
  free 
};

object * newplane(void * tex, vector ctr, vector norm) {
  plane * p;
  
  p=(plane *) rt_getmem(sizeof(plane));
  memset(p, 0, sizeof(plane));
  p->methods = &plane_methods;

  p->tex = (texture *)tex;
  p->norm = norm;
  VNorm(&p->norm);
  p->d = -VDot(&ctr, &p->norm);

  return (object *) p;
}

static int plane_bbox(void * obj, vector * min, vector * max) {
  return 0;
}

static void plane_intersect(plane * pln, ray * ry) {
  flt t,td;
   
  t=-(pln->d + VDot(&pln->norm, &ry->o));
  td=VDot(&pln->norm, &ry->d); 
  if (td != 0.0) {
    t /= td;
    if (t > 0.0)
      add_intersection(t,(object *) pln, ry);
  }
}

static void plane_normal(plane * pln, vector  * pnt, ray * incident, vector * N) {
  *N=pln->norm;
}

