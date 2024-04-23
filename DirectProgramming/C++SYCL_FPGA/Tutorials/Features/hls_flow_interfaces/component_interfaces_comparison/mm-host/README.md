# Memory-Mapped Host Interfaces
This implementation uses a register-mapped invocation interface, and demonstrates how to use `annotated_arg` to customize the memory-mapped host data interface.

![](../assets/ddr.svg)

## Invocation Interface
By default, an un-decorated oneAPI kernel will have all its control signals and arguments mapped into the IP component's control/status register (CSR).

## Data Interface - MM Host
The pointer arguments `a_in`, `b_in`, `c_out` and scalar argument `len` are passed through the IP component's CSR. In this example design, the memory-mapped host interfaces pointed to by `a_in`, `b_in`, `c_out` are customized using `annotated_arg`.

You can customize memory-mapped interfaces of your IP component if the component uses a unified shared memory (USM) host or shared pointer to access data. To customize the interface, declare your pointer arguments with the templated type `annotated_arg`.

### Declare `annotated_arg`
An explicit MM Host interface should be declared as a member of your kernel functor as shown in the  next section.

A list of properties that can be used to customize `annotated_arg` can be found in this dedicated [mmhost](/DirectProgramming/C++SYCL_FPGA/Tutorials/Features/hls_flow_interfaces/mmhost) code sample.

### Example of how to use `annotated_arg` to customize an Avalon memory-mapped host interface

```cpp
sycl::ext::oneapi::experimental::annotated_arg<
      int *, decltype(sycl::ext::oneapi::experimental::properties{
                 sycl::ext::intel::experimental::buffer_location<1>,
                 sycl::ext::intel::experimental::dwidth<32>,
                 sycl::ext::intel::experimental::latency<0>,
                 sycl::ext::intel::experimental::read_write_mode_read,
                 sycl::ext::oneapi::experimental::alignment<4>})>
      A_in;
```

## Example Output

```
Add two vectors of size 256
PASSED
```

## License
Code samples are licensed under the MIT license. See
[License.txt](/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](/third-party-programs.txt).

