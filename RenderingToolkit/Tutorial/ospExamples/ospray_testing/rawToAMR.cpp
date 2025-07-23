// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-func-template"

#include "rawToAMR.h"

#include <errno.h>
#include <stdio.h>
#include <sstream>

namespace ospray {
namespace amr {

void makeAMR(const std::vector<float> &in,
    const vec3i inGridDims,
    const int numLevels,
    const int blockSize,
    const int refinementLevel,
    const float threshold,
    std::vector<box3i> &blockBounds,
    std::vector<int> &refinementLevels,
    std::vector<float> &cellWidths,
    std::vector<std::vector<float>> &brickData)
{
  int minWidth = blockSize;

  for (int i = 1; i < numLevels; i++)
    minWidth *= refinementLevel;

  if (minWidth >= refinementLevel * reduce_max(inGridDims)) {
    throw std::runtime_error(
        "too many levels, or too fine a refinement factor."
        "do not have a single brick at the root...");
  }

  vec3i finestLevelSize = vec3i(minWidth);
  while (finestLevelSize.x < inGridDims.x)
    finestLevelSize.x += minWidth;
  while (finestLevelSize.y < inGridDims.y)
    finestLevelSize.y += minWidth;
  while (finestLevelSize.z < inGridDims.z)
    finestLevelSize.z += minWidth;

  // create container for current level so we don't use in
  std::vector<float> &currentLevel = const_cast<std::vector<float> &>(in);

  size_t numWritten = 0;
  size_t numRemoved = 0;
  std::mutex fileMutex;

  // create and write the bricks
  vec3i levelSize = finestLevelSize;
  for (int level = numLevels - 1; level >= 0; --level) {
    const vec3i nextLevelSize = levelSize / refinementLevel;
    // create container for next level down
    std::vector<float> nextLevel =
        std::vector<float>(nextLevelSize.product(), 0);

    const vec3i numBricks = levelSize / blockSize;
    rkcommon::tasking::parallel_for(numBricks.product(), [&](int brickIdx) {
      // dt == cellWidth in osp_amr_brick_info
      float dt = 1.f / powf(refinementLevel, level);
      // get 3D brick index from flat brickIdx
      const vec3i brickID(brickIdx % numBricks.x,
          (brickIdx / numBricks.x) % numBricks.y,
          brickIdx / (numBricks.x * numBricks.y));
      // set upper and lower bounds of brick based on 3D index and
      // brick size in input data space
      box3i box;
      box.lower = brickID * blockSize;
      box.upper = box.lower + (blockSize - 1);
      // current brick data
      std::vector<float> data(blockSize * blockSize * blockSize);
      size_t out = 0;
      range1f brickRange;
      // traverse the data by brick index
      for (int iz = box.lower.z; iz <= box.upper.z; iz++) {
        for (int iy = box.lower.y; iy <= box.upper.y; iy++) {
          for (int ix = box.lower.x; ix <= box.upper.x; ix++) {
            const size_t thisLevelCoord =
                ix + levelSize.y * (iy + iz * levelSize.z);
            const size_t nextLevelCoord = ix / refinementLevel
                + nextLevelSize.y
                    * (iy / refinementLevel
                        + iz / refinementLevel * nextLevelSize.z);
            // get the actual data at current coordinates
            const float v = currentLevel[thisLevelCoord];
            // insert the data into the current brick
            data[out++] = v;
            nextLevel[nextLevelCoord] +=
                v / (refinementLevel * refinementLevel * refinementLevel);
            // extend the value range of this brick (min/max) as needed
            brickRange.extend(v);
          }
        }
      }

      std::lock_guard<std::mutex> lock(fileMutex);
      if ((level > 0) && ((brickRange.upper - brickRange.lower) <= threshold)) {
        numRemoved++;
      } else {
        blockBounds.push_back(box);
        refinementLevels.push_back(level);
        cellWidths.resize(std::max(cellWidths.size(), (size_t)level + 1));
        cellWidths[level] = dt;
        brickData.push_back(data);
        numWritten++;
      }
    }); // end parallel for
    currentLevel = nextLevel;
    levelSize = nextLevelSize;
    numWritten = 0;
    numRemoved = 0;
  } // end for loop on levels
}

void outputAMR(const FileName outFileBase,
    const std::vector<box3i> &blockBounds,
    const std::vector<int> &refinementLevels,
    const std::vector<float> &cellWidths,
    const std::vector<std::vector<float>> &brickData,
    const int blockSize)
{
  // ALOK: .info is brick metadata
  FILE *infoOut = fopen(outFileBase.addExt(".info").c_str(), "wb");
  if (!infoOut) {
    throw std::runtime_error("could not open info output file!");
  }

  // ALOK: .data is actual data
  FILE *dataOut = fopen(outFileBase.addExt(".data").c_str(), "wb");
  if (!dataOut) {
    throw std::runtime_error("could not open data output file!");
  }

  std::ofstream osp(outFileBase + std::string(".osp"));
  osp << "<?xml?>" << std::endl;
  osp << "<AMRVolume>" << std::endl;
  osp << "  <fileName>" << rkcommon::FileName(outFileBase).base()
      << "</fileName>" << std::endl;
  osp << "  <brickSize>" << blockSize << "</brickSize>" << std::endl;
  osp << "  <clamp>0 100000</clamp>" << std::endl;
  osp << "</AMRVolume>" << std::endl;

  size_t numBlockBounds = blockBounds.size(),
         numRefinementLevels = refinementLevels.size(),
         numCellWidths = cellWidths.size();
  fwrite(&numBlockBounds, sizeof(size_t), 1, infoOut);
  fwrite(blockBounds.data(), sizeof(box3i), blockBounds.size(), infoOut);
  fwrite(&numRefinementLevels, sizeof(size_t), 1, infoOut);
  fwrite(
      refinementLevels.data(), sizeof(int), refinementLevels.size(), infoOut);
  fwrite(&numCellWidths, sizeof(size_t), 1, infoOut);
  fwrite(cellWidths.data(), sizeof(float), cellWidths.size(), infoOut);

  for (auto &data : brickData) {
    fwrite(data.data(), sizeof(float), data.size(), dataOut);
  }

  fclose(infoOut);
  fclose(dataOut);
}

template <typename T>
std::vector<T> loadRAW(const std::string &fileName, const vec3i &dims)
{
  const size_t num = dims.product();
  std::vector<T> volume(num);
  FILE *file = fopen(fileName.c_str(), "rb");
  if (!file)
    throw std::runtime_error(
        "ospray::amr::loadRaw(): could not open '" + fileName + "'");
  size_t numRead = fread(volume.data(), sizeof(T), num, file);
  if (num != numRead)
    throw std::runtime_error(
        "ospray::amr::loadRaw(): read incomplete data ...");
  fclose(file);

  return volume;
}

} // namespace amr
} // namespace ospray

#pragma clang diagnostic pop
