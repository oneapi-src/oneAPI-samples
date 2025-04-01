# SYCL-Graph Samples

Code examples demonstrating the usage of [`sycl_ext_oneapi_graph`](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc) extension.

| **Example**  | **Description**  |
| --------------- | --------------- |
| [Dot Product](Samples/dotProduct.cpp) | This example uses the explicit graph creation API to perform a dot product operation. |
| [Diamond Dependency](Samples/diamondDependency.cpp) | This code example shows how a SYCL queue can be put into a recording state, which allows a `command_graph` object to be populated by the command-groups submitted to the queue. Once the graph is complete, recording finishes on the queue to put it back into the default executing state. The graph is then finalized so that no more nodes can be added. Lastly, the graph is submitted in its entirety for execution via `handler::ext_oneapi_graph(command_graph<graph_state::executable>)`. |
| [Dynamic Parameter Update](Samples/dynamicParamUpdateUSM.cpp) | An example showing a graph with a single kernel node that is created using a free function kernel with `handler::set_args()` and having its node arguments updated. Additionally, [dynamicParamUpdateBuffers.cpp](Samples/dynamicParamUpdateBuffers.cpp) demonstrates using this feature with buffers and accessors. |
| [Dynamic Command Groups](Samples/dynamicCG.cpp) | Example showing how a graph with a dynamic command group node can be updated.|
| [Dynamic Command Groups With Dynamic Parameters](Samples/dynamicCG_with_Params.cpp) | Example showing how a graph with a dynamic command group that uses dynamic parameters in a node can be updated.|
| [Whole Graph Update](Samples/whole_graph_update.cpp) | Example that shows recording and updating several nodes with different parameters using whole-graph update.|

## Dependencies
The CMake configuration assumes usage of the DPC++ compiler. Both the [Intel DPC++ release](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) and the [open source version](https://github.com/intel/llvm) are compatible.

## Building
The project uses a standard CMake build configuration system. Ensure the SYCL compiler is used by the configuration either by setting the environment variable `CXX=<compiler>` or passing the configuration flag
`-DCMAKE_CXX_COMPILER=<compiler>` where `<compiler>` is your SYCL compiler's
executable (for example Intel `icpx` or LLVM `clang++`).

To check out the repository and build the examples, use simply:
```
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=<compiler>
cmake --build .
```
The CMake configuration automatically detects the available SYCL backends and
enables the SPIR/CUDA/HIP targets for the device code, including the corresponding 
architecture flags. If desired, these auto-configured cmake options may be overridden 
with the following ones:

| `<OPTION>` | `<VALUE>` |
| ---------- | ---------- |
| `ENABLE_SPIR` | `ON` or `OFF` |
| `ENABLE_CUDA` | `ON` or `OFF` |
| `ENABLE_HIP` | `ON` or `OFF` |
| `CUDA_COMPUTE_CAPABILITY` | Integer, e.g. `70` meaning capability 7.0 (arch `sm_70`) |
| `HIP_GFX_ARCH` | String, e.g. `gfx1030` |
