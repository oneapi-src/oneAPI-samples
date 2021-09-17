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
#include "devicemanager/devicemanager.hpp"
#include "pointpillars/pointpillars.hpp"
#include "pointpillars/pointpillars_config.hpp"
#include "pointpillars/pointpillars_util.hpp"

/**
 * Read in a LiDAR point cloud from a file in Point Cloud Data format (as ascii)
 * https://pointclouds.org/documentation/tutorials/pcd_file_format.html
 *
 * @param[in] file_name is the name of the PCD file
 * @param[in] points are the parsed points from the PCD as x,y,z,intensity values
 * @return number of of points in the point cloud
 */
std::size_t ReadPointCloud(std::string const &file_name, std::vector<float> &points) {
  if (!boost::filesystem::exists(file_name) || file_name.empty()) {
    return 0;
  }

  std::size_t number_of_points = 0;

  std::ifstream in(file_name);
  std::string line;
  bool parse_data = false;

  // read PCD file in a line-by-line manner
  while (std::getline(in, line) && points.size() <= 4 * number_of_points) {
    if (parse_data) {
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
      number_of_points = atoll(line.substr(7).c_str());
    } else if (line.find("DATA") != std::string::npos) {
      parse_data = true;
    }
  }

  return number_of_points;
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

  // parse program options
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("list")) {
    devicemanager::GetDevices();
    return 1;
  }

  std::vector<sycl::info::device_type> execution_devices;

  if (vm.count("cpu")) {
    execution_devices.push_back(sycl::info::device_type::cpu);
  }

  if (vm.count("gpu")) {
    execution_devices.push_back(sycl::info::device_type::gpu);
  }

  if ((vm.count("host")) || execution_devices.empty()) {
    execution_devices.push_back(sycl::info::device_type::host);
  }

  // Point Pillars initialization
  pointpillars::PointPillarsConfig config;
  std::vector<pointpillars::ObjectDetection> object_detections;

  // read point cloud
  std::size_t number_of_points;
  std::vector<float> points;
  number_of_points = ReadPointCloud("example.pcd", points);

  // if the point cloud was empty, something went wrong
  if ((number_of_points == 0) || points.empty()) {
    std::cout << "Unable to read point cloud file. Please put the point cloud file into the data/ folder." << std::endl;
    return -1;
  }

  // Run PointPillars for each execution device
  for (const auto &device_type : execution_devices) {
    if (!devicemanager::SelectDevice(device_type)) {
      std::cout << "\n\n";
      continue;
    }

    // setup PointPillars
    pointpillars::PointPillars point_pillars(0.5f, 0.5f, config);
    const auto start_time = std::chrono::high_resolution_clock::now();

    // run PointPillars
    try {
      point_pillars.Detect(points.data(), number_of_points, object_detections);
    } catch (...) {
      std::cout << "Exception during PointPillars execution\n";
      return -1;
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms\n\n";

    // print results
    std::cout << object_detections.size() << " cars detected\n";

    for (auto const &detection : object_detections) {
      std::cout << config.classes[detection.class_id] << ": Probability = " << detection.class_probabilities[0]
                << " Position = (" << detection.x << ", " << detection.y << ", " << detection.z
                << ") Length = " << detection.length << " Width = " << detection.width << "\n";
    }
    std::cout << "\n\n";
  }

  return 0;
}
