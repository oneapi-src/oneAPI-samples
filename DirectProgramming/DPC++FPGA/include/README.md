# Include for Shared Header Libraries
This directory contains utility header libraries optimized for SYCL*-compliant FPGA designs. Find usage examples in the **ReferenceDesigns** and **Tutorials** directories.

## Available Header Libraries

### Utilities

| Filename                     | Description
---                            |---
| `constexpr_math.hpp`           | Defines utilities for statically computing math functions (for example, Log2 and Pow2).
| `memory_utils.hpp`             | Generic functions for streaming data from memory to a SYCL pipe and vise versa.
| `metaprogramming_utils.hpp`    | Defines various metaprogramming utilities (for example, generating a power of 2 sequence and checking if a type has a subscript operator).
| `onchip_memory_with_cache.hpp` | Class that contains an on-chip memory array with a register backed cache to achieve high performance read-modify-write loops.
| `pipe_utils.hpp`               | Utility classes for working with pipes, such as PipeArray.
| `rom_base.hpp`                 | A generic base class to create ROMs in the FPGA using and initializer lambda or functor.
| `tuple.hpp`                    | Defines a template to implement tuples.
| `unrolled_loop.hpp`            | Defines a templated implementation of unrolled loops.

### Linear Algebra

| Filename               | Description
---                      |---
| `streaming_qrd.hpp`      | QR decomposition of matrices with pipe interfaces.
| `streaming_qri.hpp`      | QR-based inversion of matrices with pipe interfaces.
| `streaming_cholesky.hpp` | Cholesky decomposition of matrices with pipe interfaces.
| `streaming_cholesky_inversion.hpp` | Cholesky-based inversion of matrices with pipe interfaces.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).