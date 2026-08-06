// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rkcommon/math/box.h"
#include "rkcommon/os/FileName.h"
#include "rkcommon/tasking/parallel_for.h"

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <fcntl.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ospray {
namespace amr {

using namespace rkcommon;
using namespace rkcommon::math;

void makeAMR(const std::vector<float> &in,
    const vec3i inGridDims,
    const int numLevels,
    const int blockSize,
    const int refinementLevel,
    const float threshold,
    std::vector<box3i> &blockBounds,
    std::vector<int> &refinementLevels,
    std::vector<float> &cellWidths,
    std::vector<std::vector<float>> &brickData);

void outputAMR(const FileName outFileBase,
    const std::vector<box3i> &blockBounds,
    const std::vector<int> &refinementLevels,
    const std::vector<float> &cellWidths,
    const std::vector<std::vector<float>> &brickData,
    const int blockSize);

template <typename T>
std::vector<T> loadRAW(const std::string &fileName, const vec3i &dims);

template <typename T>
std::vector<T> mmapRAW(const std::string &fileName, const vec3i &dims)
{
#ifdef _WIN32
  throw std::runtime_error("Memory mapping not supported in Windows");
#else
  FILE *file = fopen(fileName.c_str(), "rb");
  if (!file)
    throw std::runtime_error("Could not open file.");
  fseek(file, 0, SEEK_END);
  size_t actualFileSize = ftell(file);
  fclose(file);

  size_t fileSize =
      size_t(dims.x) * size_t(dims.y) * size_t(dims.z) * sizeof(T);

  if (actualFileSize != fileSize) {
    std::stringstream ss;
    ss << "Got file size " << prettyNumber(actualFileSize);
    ss << ", but expected " << prettyNumber(fileSize);
    ss << ". Common cause is wrong data type!";
    throw std::runtime_error(ss.str());
  }

  int fd = ::open(fileName.c_str(), O_RDONLY);
  if (fd == -1)
    throw std::runtime_error("Could not open file.");

  void *mem = mmap(nullptr,
      fileSize,
      PROT_READ,
      MAP_SHARED // |MAP_HUGETLB
      ,
      fd,
      0);
  ::close(fd);
  if (mem == MAP_FAILED)
    throw std::runtime_error("Could not mmap file.");

  std::vector<T> volume;
  volume.assign((T *)mem, (T *)mem + dims.product());

  return volume;
#endif
}

} // namespace amr
} // namespace ospray
