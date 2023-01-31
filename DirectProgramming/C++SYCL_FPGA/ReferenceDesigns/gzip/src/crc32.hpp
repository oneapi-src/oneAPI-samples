#ifndef __CRC32_H__
#define __CRC32_H__
#pragma once

#include <stdint.h>
#include <stdlib.h>

uint32_t Crc32Host(
    const char *pbuf,        // pointer to the buffer to crc
    size_t sz,               // number of bytes
    uint32_t previous_crc);  // previous CRC, allows combining. First invocation
                             // would use 0xffffffff.
uint32_t Crc32(const char *pbuf,        // pointer to the buffer to crc
               size_t sz,               // number of bytes
               uint32_t previous_crc);  // previous CRC, allows combining. First
                                        // invocation would use 0xffffffff.

#endif  //__CRC32_H__
