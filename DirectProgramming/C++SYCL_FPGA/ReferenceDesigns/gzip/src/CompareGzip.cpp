#include "CompareGzip.hpp"
#include <sys/stat.h>

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
  mode_t mask = umask(S_IXUSR);
  mkstemp(tmp_name);
  umask(mask);
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
