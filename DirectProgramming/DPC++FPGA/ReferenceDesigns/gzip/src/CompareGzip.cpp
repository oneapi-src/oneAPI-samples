// ==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of
// California and by the laws of the United States of America.

#include "CompareGzip.hpp"

// returns 0 on success, otherwise failure
int CompareGzipFiles(
    const std::string
        &original_file,  // original input file to compare gzip uncompressed
    const std::string &input_gzfile)  // gzip file to check
{
#ifdef _MSC_VER
  std::cout
      << "Info: skipping output verification on Windows, no builtin gunzip\n";
  return 0;
#else
  //------------------------------------------------------------------
  // assume all good to start with.

  int gzipstatus = 0;

  //------------------------------------------------------------------
  // Create temporary output filename for gunzip

  char tmp_name[] = "/tmp/gzip_fpga.XXXXXX";
  mkstemp(tmp_name);
  std::string outputfile = tmp_name;

  //------------------------------------------------------------------
  // Check that the original file and gzipped file exist.

  //------------------------------------------------------------------
  // gunzip the file produced to stdout, capturing to the temp file.

  std::string cmd = "gunzip -c ";
  cmd += input_gzfile;
  cmd += " > " + outputfile;

  int gzout = ::system(cmd.c_str());
  if (gzout != 0) {
    gzipstatus = 3;
  }

  //------------------------------------------------------------------
  // diff the temp file and the original.

  cmd = "diff -q " + outputfile + " " + original_file;
  int diffout = ::system(cmd.c_str());
  if (diffout != 0) {
    gzipstatus = 4;
  }

  //------------------------------------------------------------------
  // Cleanup, remove the temp file.

  (void)::remove(outputfile.c_str());

  return gzipstatus;
#endif
}
