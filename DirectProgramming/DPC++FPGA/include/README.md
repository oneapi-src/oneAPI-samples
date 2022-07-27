# Include for Shared Header Libraries
This directory contains utility header libraries optimized for FPGA DPC++ designs. Examples of their usage can be found in ReferenceDesigns and Tutorials.

## Available Header Libraries

### Utilities

| Filename                     | Description                                                                                                                     | Use case examples
---                            |---                                                                                                                              |---
| constexpr_math.hpp           | Defines utilities for statically computing math functions (e.g. Log2 and Pow2).                                                 | `ReferenceDesigns/merge_sort/`, `ReferenceDesigns/qrd`, `ReferenceDesigns/qri`
| memory_utils.hpp             | Generic functions for streaming data from memory to a SYCL pipe, and vice versa.                                                | `ReferenceDesigns/decompress/`
| metaprogramming_utils.hpp    | Defines various metapgramming utilities (e.g. generating a power of 2 sequence and checking if a type has a subscript operator).| `ReferenceDesigns/decompress/`, `include/unrolled_loop.hpp`
| onchip_memory_with_cache.hpp | Class that contains an on-chip memory array with a register backed cache to achieve high performance read-modify-write loops.   | `Tutorials/DesignPatterns/onchip_memory_cache/`, ReferenceDesigns/decompress/`, `ReferenceDesigns/db/`
| pipe_utils.hpp               | Utility classes for working with pipes, such as PipeArray.                                                                      | `Tutorials/DesignPatterns/pipe_array/`, `ReferenceDesigns/merge_sort/`, `ReferenceDesigns/gzip/`, `ReferenceDesigns/mvdr_beamforming/`
| rom_base.hpp                 | A generic base class to create ROMs in the FPGA using and initializer lambda or functor.                                        | `ReferenceDesigns/anr/`
| tuple.hpp                    | Defines a template to implement tuples.                                                                                         | `ReferenceDesigns/cholesky_inversion/`, `ReferenceDesigns/qri/`, `ReferenceDesigns/cholesky/`
| unrolled_loop.hpp            | Defines a templated implementation of unrolled loops.                                                                           | `Tutorials/DesignPatterns/pipe_array/`, `ReferenceDesigns/cholesky/`, `ReferenceDesigns/anr/`

### Linear Algebra

| Filename               | Description                                                             | Use case examples
---                      |---                                                                      |---
| streaming_qrd.hpp      | QR decomposition of matrices with pipe interfaces.                      | `ReferenceDesigns/qrd`
| streaming_qri.hpp      | QR-based inversion of matrices with pipe interfaces.                    | `ReferenceDesigns/qri`
| streaming_cholesky.hpp | Cholesky decomposition of matrices with pipe interfaces.                | `ReferenceDesigns/cholesky`
| streaming_cholesky_inversion.hpp | Cholesky-based inversion of matrices with pipe interfaces.    | `ReferenceDesigns/cholesky_inversion`

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
