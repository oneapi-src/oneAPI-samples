//==============================================================
// Copyright © 2022 - 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "../../common/aspect_queries.hpp"

#include <cstddef>
#include <iostream>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

namespace sycl_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

int main() {
  constexpr size_t Size = 10;
  constexpr size_t Iter = 5;

  float Alpha = 1.0f;
  float Beta = 2.0f;
  float Gamma = 3.0f;

  queue Queue{};

  ensure_graph_support(Queue.get_device());

  sycl_ext::command_graph Graph(Queue.get_context(), Queue.get_device());

  float *Dotp = malloc_device<float>(1, Queue);
  float *X = malloc_device<float>(Size, Queue);
  float *Y = malloc_device<float>(Size, Queue);
  float *Z = malloc_device<float>(Size, Queue);

  // Add commands to the graph to create the following topology.
  //
  //     i
  //    / \
  //   a   b
  //    \ /
  //     c

  // Init data on the device.
  auto Node_i = Graph.add([&](handler &CGH) {
    CGH.parallel_for(Size, [=](id<1> Id) {
      const size_t i = Id[0];
      X[i] = 1.0f;
      Y[i] = 3.0f;
      Z[i] = 2.0f;
      *Dotp = 0.;
    });
  });

  auto Node_a = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>{Size}, [=](id<1> Id) {
          const size_t i = Id[0];
          X[i] = Alpha * X[i] + Beta * Y[i];
        });
      },
      {sycl_ext::property::node::depends_on(Node_i)});

  auto Node_b = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>{Size}, [=](id<1> Id) {
          const size_t i = Id[0];
          Z[i] = Gamma * Z[i] + Beta * Y[i];
        });
      },
      {sycl_ext::property::node::depends_on(Node_i)});

  auto Node_c = Graph.add(
      [&](handler &CGH) {
        CGH.single_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            *Dotp += X[i] * Z[i];
          }
        });
      },
      {sycl_ext::property::node::depends_on(Node_a, Node_b)});

  auto GraphExec = Graph.finalize();

  // Use queue shortcut for graph submission.
  for (int i = 0; i < Iter; ++i) {
    Queue.ext_oneapi_graph(GraphExec).wait();
  }

  // Copy the result back to the host.
  float ResultDotp;
  Queue.copy(Dotp, &ResultDotp, 1).wait();

  // Verify the results.
  float HostDotp{0.0f};
  std::vector<float> HostX(Size, 1.0f), HostY(Size, 3.0f), HostZ(Size, 2.0f);
  for (size_t i = 0; i < Size; ++i) {
    HostX[i] = Alpha * HostX[i] + Beta * HostY[i];
    HostZ[i] = Gamma * HostZ[i] + Beta * HostY[i];
    HostDotp += HostX[i] * HostZ[i];
  }

  assert(HostDotp == ResultDotp);

  std::cout << "Success!" << std::endl;

  // Memory is freed outside the graph.
  free(X, Queue);
  free(Y, Queue);
  free(Z, Queue);
  free(Dotp, Queue);

  return 0;
}
