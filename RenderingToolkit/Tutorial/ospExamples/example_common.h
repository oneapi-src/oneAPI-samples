// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ospray
#include "ospray/ospray.h"
// std
#include <iostream>
#include <stdexcept>

inline void initializeOSPRay(
    int argc, const char **argv, bool errorsFatal = true)
{
  // initialize OSPRay; OSPRay parses (and removes) its commandline parameters,
  // e.g. "--osp:debug"
  OSPError initError = ospInit(&argc, argv);

  if (initError != OSP_NO_ERROR)
    throw std::runtime_error("OSPRay not initialized correctly!");

  OSPDevice device = ospGetCurrentDevice();
  if (!device)
    throw std::runtime_error("OSPRay device could not be fetched!");

  // set an error callback to catch any OSPRay errors and exit the application
  if (errorsFatal) {
    ospDeviceSetErrorCallback(
        device,
        [](void *, OSPError error, const char *errorDetails) {
          std::cerr << "OSPRay error: " << errorDetails << std::endl;
          exit(error);
        },
        nullptr);
  } else {
    ospDeviceSetErrorCallback(
        device,
        [](void *, OSPError, const char *errorDetails) {
          std::cerr << "OSPRay error: " << errorDetails << std::endl;
        },
        nullptr);
  }

  ospDeviceSetStatusCallback(
      device, [](void *, const char *msg) { std::cout << msg; }, nullptr);

  bool warnAsErrors = true;
  auto logLevel = OSP_LOG_WARNING;

  ospDeviceSetParam(device, "warnAsError", OSP_BOOL, &warnAsErrors);
  ospDeviceSetParam(device, "logLevel", OSP_INT, &logLevel);

  ospDeviceCommit(device);
  ospDeviceRelease(device);
}
