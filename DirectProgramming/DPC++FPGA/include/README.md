# Include for Shared Header Libraries
This directory contains utility header libraries optimized for FPGA DPC++ designs. Examples of their usage can be found in ReferenceDesigns and Tutorials.

## Available Header Libraries

### Utilities

| Filename       | Description
---              |---
| rom_base.hpp | A generic base class to create ROMs in the FPGA using and initializer lambda or functor.
| constexpr_math.hpp | Defines utilities for statically computing math functions such as Log2.
| pipe_utils.hpp | Utility classes for working with pipes, such as PipeArray.
| tuple.hpp | Defines a template to implement tuples.
| unrolled_loop.hpp | Defines a templated implementation of unrolled loops.

### Linear algebra

| Filename       | Description
---              |---
| streaming_qrd.hpp | QR decomposition of matrices with pipe interfaces.
| streaming_qri.hpp | QR-based inversion of matrices with pipe interfaces.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

