//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iostream>

// Custom device selector to select a device of the specified type.
// The platform of the device has to contain the phrase "Intel".  If
// the platform or the type are not as expected, the queried device is
// not selected, as indicated by a negative score.

using namespace std;
using namespace sycl;

class CustomSelector : public device_selector {
 public:
  CustomSelector(info::device_type type) : type{type} {}

  int operator()(const device& dev) const override {
    if (type != dev.get_info<info::device::device_type>()) return -1;

    string platform_name =
        dev.get_platform().get_info<info::platform::name>();

    if (platform_name.find("Intel") != string::npos)
      return 10;
    else
      return (type == info::device_type::host) ? 10 : -1;
  }

 private:
  info::device_type type;
};

// Return the device type based on the program arguments.

static info::device_type GetDeviceType(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " "
         << "<host|cpu|gpu|accelerator>\n";
    exit(1);
  }

  string type_arg{argv[1]};
  info::device_type type;

  if (type_arg.compare("host") == 0)
    type = info::device_type::host;
  else if (type_arg.compare("cpu") == 0)
    type = info::device_type::cpu;
  else if (type_arg.compare("gpu") == 0)
    type = info::device_type::gpu;
  else if (type_arg.compare("accelerator") == 0)
    type = info::device_type::accelerator;
  else {
    cerr << "fail; unrecognized device type '" << type_arg << "'\n";
    exit(-1);
  }

  return type;
}