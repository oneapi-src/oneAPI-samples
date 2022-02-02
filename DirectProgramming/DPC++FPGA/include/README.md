# Include for Shared Header Libraries
This directory contains utility header libraries optimized for FPGA DPC++ designs. Examples of their usage can be found in ReferenceDesigns and Tutorials.

## Available Header Libraries
| Filename       | Description
---              |---
| pipe_utils.hpp | Defines utilities for working with pipes, such as PipeArray.
| metaprogramming_math.hpp | Defines utilities for statically computing math functions such as Log2.
| tuple.hpp | Defines a template to implement tuples.
| unrolled_loop.hpp | Defines a templated implementation of unrolled loops.
| streaming_qrd.hpp | Defines a functor that implements the QR decomposition of matrices.
| streaming_qri.hpp | Defines a functor that implements the QR-based inversion of matrices.
| utils.hpp | Defines commonly used classes.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

