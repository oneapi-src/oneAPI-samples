/*
 *
 *
 * Copyright (C) 2005 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials, and your use of them
 * is governed by the express license under which they were provided to you ("License"). Unless
 * the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose
 * or transmit this software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef WIN32
#include <windows.h>

int getCPUCount()
{
	int processorCount = 0;
	// Get the mask of available processors for this process.
	DWORD_PTR ProcessAffinityMask;
	DWORD_PTR SystemAffinityMask;
	DWORD_PTR ProcessorBit;

	if (GetProcessAffinityMask(GetCurrentProcess(), &ProcessAffinityMask, &SystemAffinityMask)) {
		// Check each bit in the mask for an available processor.
		for (ProcessorBit = 1; ProcessorBit > 0; ProcessorBit <<= 1) {
			// Increase the processor count
			if (ProcessAffinityMask & ProcessorBit)
				processorCount++;
		}
	}
	return processorCount;
}
#else
#include <unistd.h>
/*-------------------------------------------------
 * gets CPU freqeuency in Hz (Linux only)
 * from /proc/cpuinfo
 *------------------------------------------------*/

double getCPUFreq() {
   #define BUFLEN 110

   FILE* sysinfo;
   char* ptr;
   char buf[BUFLEN];
   char key[] = "cpu MHz";
   int keylen = sizeof( key ) - 1;
   double freq = -1;

   sysinfo = fopen( "/proc/cpuinfo", "r" );
   if( sysinfo != NULL ) {
      while( fgets( buf, BUFLEN, sysinfo ) != NULL ) {
         if( !strncmp( buf, key, keylen ) ) {
            ptr = strstr( buf, ":" );
            freq = atof( ptr+1 ) * 1000000;
            break;
         }
      }
      fclose( sysinfo );
   }
   fprintf(stderr, "Freq = %f GHz\n", freq / 1000000000);
   return freq;
}

int getCPUCount() {
	return sysconf(_SC_NPROCESSORS_CONF);
}

#endif
