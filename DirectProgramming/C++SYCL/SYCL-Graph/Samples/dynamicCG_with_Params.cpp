/* Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the Apache License, Version 2.0.
 * For a copy, see http://www.apache.org/licenses/LICENSE-2.0
 */

#include "common/aspect_queries.hpp"

#include <sycl/sycl.hpp>

namespace sycl_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

constexpr size_t Size = 1024;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl_ext::single_task_kernel))
void ff_0(int *Ptr) {
  for (size_t i{0}; i < Size; ++i) {
    Ptr[i] = i;
  }
}

int main() {
  queue Queue{};

  ensure_full_graph_support(Queue.get_device());

  auto Context = Queue.get_context();
  auto Device = Queue.get_device();

  sycl_ext::command_graph Graph{Context, Device};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);

  // Create  dynamic parameters with the initial values: PtrA
  sycl_ext::dynamic_parameter DynParamPtr(Graph, PtrA);

#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(Context);
  kernel_id Kernel_id = sycl_ext::get_kernel_id<ff_0>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);

  auto CGFA = [&](handler &CGH) {
    CGH.set_arg(0, DynParamPtr);
    CGH.single_task(Kernel);
  };

  // Kernel has a single argument, a dynamic parameter of ptr type
  auto CGFB = [&](handler &CGH) {
    CGH.set_arg(0, DynParamPtr);
    CGH.single_task(Kernel);
  };

  // Construct a dynamic command-group with CGFA as the active cgf (index 0).
  auto DynamicCG = sycl_ext::dynamic_command_group(Graph, {CGFA, CGFB});

  // Create a dynamic command-group graph node
  auto DynamicCGNode = Graph.add(DynamicCG);

  auto ExecGraph = Graph.finalize(sycl_ext::property::graph::updatable{});

  // The graph will execute CGFA with PtrA.
  Queue.ext_oneapi_graph(ExecGraph).wait();

  // Update DynParamPtr with a new value
  DynParamPtr.update(PtrB);

  // Sets CGFB as active in the dynamic command-group (index 1).
  DynamicCG.set_active_index(1);

  // Calls update to update the executable graph node with the changes to
  // DynamicCG and DynParamPtr.
  ExecGraph.update(DynamicCGNode);

  // The graph will execute CGFB with PtrB.
  Queue.ext_oneapi_graph(ExecGraph).wait();

#endif
  sycl::free(PtrA, Queue);
  sycl::free(PtrB, Queue);

  return 0;
}
