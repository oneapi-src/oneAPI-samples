// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Builder.h"

namespace ospray {
namespace testing {

template <typename T>
inline void setParam(
    SceneBuilderHandle _b, const std::string &type, const T &val)
{
  auto *b = (detail::Builder *)_b;
  b->setParam(type, val);
}

} // namespace testing
} // namespace ospray
