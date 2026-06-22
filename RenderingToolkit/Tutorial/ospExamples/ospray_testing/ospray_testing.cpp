// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ospray_testing
#include "ospray_testing.h"
// stl
#include <map>

using namespace rkcommon;
using namespace rkcommon::math;

namespace ospray {
namespace testing {

SceneBuilderHandle newBuilder(const std::string &type)
{
  auto *b = detail::Builder::createBuilder(type);
  if (b == nullptr)
    throw std::runtime_error("Unable to find '" + type + "' builder");
  return (SceneBuilderHandle)b;
}

cpp::Group buildGroup(SceneBuilderHandle _b)
{
  auto *b = (detail::Builder *)_b;
  return b->buildGroup();
}

cpp::World buildWorld(SceneBuilderHandle _b)
{
  auto *b = (detail::Builder *)_b;
  return b->buildWorld();
}

void commit(SceneBuilderHandle _b)
{
  auto *b = (detail::Builder *)_b;
  b->commit();
}

void release(SceneBuilderHandle _b)
{
  auto *b = (detail::Builder *)_b;
  b->refDec();
}

} // namespace testing
} // namespace ospray
