//=======================================================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// ======================================================================================
#pragma pack(push, 1)

// This is the data structure which is going to represent one pixel value in RGB
// format
typedef struct {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
} rgb;

// This block is only used when build for Structure of Arays (SOA) with Array
// Notation
typedef struct {
  unsigned char *blue;
  unsigned char *green;
  unsigned char *red;
} SOA_rgb;

#pragma pack(pop)
