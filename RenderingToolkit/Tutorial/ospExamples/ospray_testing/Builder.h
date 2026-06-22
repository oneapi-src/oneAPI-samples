// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ospray
#include "ospray/ospray_cpp.h"
#include "ospray/ospray_cpp/ext/rkcommon.h"
// rkcommon
#include "rkcommon/memory/IntrusivePtr.h"
#include "rkcommon/utility/ParameterizedObject.h"
// std
#include <functional>
#include <map>

#include "ospray_testing.h"

namespace ospray {
namespace testing {
namespace detail {

using namespace rkcommon;
using namespace rkcommon::math;

struct Builder : public memory::RefCountedObject,
                 public utility::ParameterizedObject
{
  using BuilderFcn = std::function<Builder *()>;

  ~Builder() override = default;

  virtual void commit();

  virtual cpp::Group buildGroup() const = 0;
  virtual cpp::World buildWorld() const;
  virtual cpp::World buildWorld(
      const std::vector<cpp::Instance> &instances) const;

  static void registerBuilder(const std::string &name, BuilderFcn fcn);
  static Builder *createBuilder(const std::string &name);

 protected:
  cpp::TransferFunction makeTransferFunction(const range1f &valueRange) const;

  cpp::Instance makeGroundPlane(const box3f &bounds) const;

  // Data //

  std::string rendererType{"scivis"};
  std::string tfColorMap{"jet"};
  std::string tfOpacityMap{"linear"};

  bool addPlane{true};

  unsigned int randomSeed{0};

 private:
  using BuilderFactory = std::map<std::string, BuilderFcn>;

  static std::unique_ptr<BuilderFactory> factory;
};

} // namespace detail
} // namespace testing
} // namespace ospray

#define OSP_REGISTER_TESTING_BUILDER(InternalClassName, Name)                  \
  static bool init_builder_##Name()                                            \
  {                                                                            \
    using Builder = ospray::testing::detail::Builder;                          \
    Builder::registerBuilder(                                                  \
        TOSTRING(Name), [] { return new InternalClassName; });                 \
    return true;                                                               \
  }                                                                            \
  static bool val_##Name##_initialized = init_builder_##Name();
