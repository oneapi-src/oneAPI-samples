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
 * macros.h - This file contains macro versions of functions that would be best 
 * used as inlined code rather than function calls.
 *
 *  $Id: macros.h,v 1.2 2007-02-22 17:54:15 Exp $
 */

#define MYMAX(a , b) ((a) > (b) ? (a) : (b))
#define MYMIN(a , b) ((a) < (b) ? (a) : (b))

#define VDOT(return, a, b) 				\
 return=(a.x * b.x  +  a.y * b.y  +  a.z * b.z); 	\

#define RAYPNT(c, a, b)		\
c.x = a.o.x + ( a.d.x * b );	\
c.y = a.o.y + ( a.d.y * b );	\
c.z = a.o.z + ( a.d.z * b );	\


#define VSUB(a, b, c)		\
c.x = (a.x - b.x);		\
c.y = (a.y - b.y);		\
c.z = (a.z - b.z);		\


#define VCROSS(a, b, c) 				\
 c->x = (a->y * b->z) - (a->z * b->y);			\
 c->y = (a->z * b->x) - (a->x * b->z);			\
 c->z = (a->x * b->y) - (a->y * b->x);			\

