//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
template <size_t it, size_t end> struct Unroller {
  template <typename Action> static void Step(const Action &action) {
    action(std::integral_constant<size_t, it>());
    Unroller<it + 1, end>::Step(action);
  }
};

template <size_t end> struct Unroller<end, end> {
  template <typename Action> static void Step(const Action &) {}
};
