#ifndef __COMPAREGZIP_H__
#define __COMPAREGZIP_H__
#pragma once

#include <iostream>
#include <string>

int CompareGzipFiles(
    const std::string
        &original_file,  // original input file to compare gzip uncompressed
    const std::string &input_gzfile);  // gzip file to check

#endif  //__COMPAREGZIP_H__
