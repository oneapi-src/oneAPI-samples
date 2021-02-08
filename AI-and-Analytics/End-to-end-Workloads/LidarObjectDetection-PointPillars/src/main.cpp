//==============================================================
// Copyright © 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/range/iterator_range.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include "PointPillars/PointPillarsConfig.hpp"
#include "PointPillars/PointPillarsUtil.hpp"
#include "PointPillars/inference/pointpillars.hpp"
#include "PointPillars/operations/common.hpp"

std::size_t readPointCloud(std::string const &fileName, std::vector<float> &points) {
  if (!boost::filesystem::exists(fileName) || fileName.empty()) {
    return 0;
  }

  std::size_t numberOfPoints = 0;

  std::ifstream in(fileName);
  std::string line;
  bool parseData = false;
  while (std::getline(in, line) && points.size() <= 4 * numberOfPoints) {
    if (parseData) {
      std::istringstream iss(line);
      float x, y, z, intensity;
      double timestamp;
      if (!(iss >> x >> y >> z >> intensity >> timestamp)) {
        return 0;
      }
      points.push_back(x);
      points.push_back(y);
      points.push_back(z);
      points.push_back(intensity);
    } else if (line.find("POINTS") != std::string::npos) {
      numberOfPoints = atoll(line.substr(7).c_str());
    } else if (line.find("DATA") != std::string::npos) {
      parseData = true;
    }
  }

  return numberOfPoints;
}

int main(int argc, char *argv[]) {
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help", "produce help message")
    ("host", "Use single-threaded CPU as execution device (default)")
    ("cpu", "Use CPU as execution device")
    ("gpu", "Use GPU as execution device")
    ("list", "Get available execution devices");
  // clang-format on

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("list")) {
    getSyclDevices();
    return 1;
  }

  std::vector<cl::sycl::info::device_type> executionDevices;

  if (vm.count("cpu")) {
    executionDevices.push_back(cl::sycl::info::device_type::cpu);
  }

  if (vm.count("gpu")) {
    executionDevices.push_back(cl::sycl::info::device_type::gpu);
  }

  if ((vm.count("host")) || executionDevices.empty()) {
    executionDevices.push_back(cl::sycl::info::device_type::host);
  }

  // Point Pillars initialization
  dnn::PointPillarsConfig config;
  std::vector<dnn::ObjectDetection> objectDetections;

  // read point cloud
  std::size_t numberOfPoints;
  std::vector<float> points;
  numberOfPoints = readPointCloud("example.pcd", points);

  if ((numberOfPoints == 0) || points.empty()) {
    std::cout << "Unable to read point cloud file. Please put the point cloud file into the data/ folder." << std::endl;
    return 1;
  }

  for (const auto &device : executionDevices) {
    if (!changeDefaultSyclDevice(device)) {
      std::cout << "\n\n";
      continue;
    }

    dnn::PointPillars pointPillars(0.5f, 0.5f, config);

    std::chrono::high_resolution_clock::time_point tStartOneAPI = std::chrono::high_resolution_clock::now();
    try {
      pointPillars.detect(points.data(), numberOfPoints, objectDetections);
    } catch (...) {
      std::cout << "Exception during PointPillars execution\n";
    }
    std::chrono::high_resolution_clock::time_point tEndOneAPI = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tEndOneAPI - tStartOneAPI).count() << "ms\n\n";

    std::cout << objectDetections.size() << " cars detected\n";

    for (auto const &detection : objectDetections) {
      std::cout << config.classes[detection.classId] << ": Probability = " << detection.classProbabilities[0]
                << " Position = (" << detection.x << ", " << detection.y << ", " << detection.z
                << ") Length = " << detection.length << " Width = " << detection.width << "\n";
    }
    std::cout << "\n\n";
  }

  return 0;
}