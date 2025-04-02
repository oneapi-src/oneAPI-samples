// ==============================================================
// Copyright © 2025 Codeplay Software
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "common/aspect_queries.hpp"

#include <sycl/sycl.hpp>

namespace sycl_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl_ext::nd_range_kernel<1>))
void ff_1(int *PtrX, int *PtrY, int *PtrZ) {
  size_t GlobalID =
      ext::oneapi::this_work_item::get_nd_item<1>().get_global_id();
  PtrX[GlobalID] += PtrY[GlobalID] * PtrZ[GlobalID];
}

int main() {
  constexpr size_t Size = 1024;

  queue Queue{};

  ensure_full_aspects_support(Queue.get_device());

  auto Context = Queue.get_context();
  auto Device = Queue.get_device();

  // USM allocations for kernel input/output
  int *PtrX = malloc_shared<int>(Size, Queue);
  int *PtrY = malloc_device<int>(Size, Queue);

  int *PtrZ = malloc_shared<int>(Size, Queue);
  int *PtrQ = malloc_device<int>(Size, Queue);

#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(Context);
  kernel_id Kernel_id = sycl_ext::get_kernel_id<ff_1>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);

  // Graph containing a kernel node
  sycl_ext::command_graph Graph(Context, Device);

  int Scalar = 42;
  // Create graph dynamic parameters
  sycl_ext::dynamic_parameter DynParamInput(Graph, PtrX);
  sycl_ext::dynamic_parameter DynParamScalar(Graph, Scalar);

  // The node uses PtrX as an input & output parameter, with operand
  // mySclar as another argument.
  sycl_ext::node KernelNode = Graph.add([&](handler &CGH) {
    CGH.set_args(DynParamInput, PtrY, DynParamScalar);
    CGH.parallel_for(range{Size}, Kernel);
  });

  // Create an executable graph with the updatable property.
  auto ExecGraph = Graph.finalize(sycl_ext::property::graph::updatable{});

  // Execute graph, then update without needing to wait for it to complete
  Queue.ext_oneapi_graph(ExecGraph);

  // Change ptrX argument to PtrZ
  DynParamInput.update(PtrZ);

  // Change Scalar argument to NewScalar
  int NewScalar = 12;
  DynParamScalar.update(NewScalar);

  // Update KernelNode in the executable graph with the new parameters
  ExecGraph.update(KernelNode);
  // Execute graph again
  Queue.ext_oneapi_graph(ExecGraph);
  Queue.wait();
#endif

  sycl::free(PtrX, Queue);
  sycl::free(PtrY, Queue);
  sycl::free(PtrZ, Queue);
  sycl::free(PtrQ, Queue);
}
