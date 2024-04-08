# oneAPI SYCL HLS Gaskets

The two Platform Designer IPs in this repository simplify the process of using oneAPI video IPs with the existing [Intel Video and Vision Processing (VVP)](https://www.intel.com/content/www/us/en/products/details/fpga/intellectual-property/dsp/video-vision-processing-suite.html) IP suite.

These gaskets should be placed on either side of a oneAPI IP to adapt the AXI4-Streaming physical layer used by the VVP IPs to the Avalon Streaming physical layer used by the oneAPI IPs. 

![oneAPI gaskets flanking a oneAPI streaming IP](assets/schematic.png)

## Description

These gaskets do NOT do any protocol conversion, they simply route the sideband signals as follows:

| AXI4-Streaming Signal | Avalon Streaming Signal |
|-----------------------|-------------------------|
| `tready`              | `ready`                 |
| `tvalid`              | `valid`                 |
| `tdata`               | `data`                  |
| `tlast`               | `endofpacket`           |
| `tuser[0]`            | `startofpacket`         |

Furthermore, these gaskets also adapt the padding between pixels used in the VVP video protocol to match the padding requirements imposed by limitations of SYCL HLS. SYCL HLS inserts padding bits after each *color channel*, while the AXI4-Streaming video protocol requires padding only be inserted after each *pixel*. The IPs in this repository adapt the pixels appropriately.

**oneAPI Padding**

![](assets/SYCL_HLS_padding.png)


**VVP Padding**

![](assets/VVP_Padding.png)

## Usage

You can add these IPs to a Platform Designer system easily. When you instantiate an IP, you will see a parameter editor screen like this:

![](assets/parameters.png)

Set the `Parallel Pixels`, `Color Channels`, and `Bits per Channel` properties to match the parameterizations of your VVP IPs. The Avalon interfaces will automatically be calculated. You can view the intermediate calculations by ticking the `Show Derived Parameters` check box, or by inspecting the `Signals and Interfaces` tab under `Component Instantiation`.

![](assets/signals-and-interfaces.png)

Here is an example of how to connect up the IPs in the Platform Designer patch panel:

![](assets/patch-panel.png)

## Testing

This IP includes a simple testbench that you can run with Questasim using the included `test.do` script:

```
vsim -c -do test_avalon_to_axi.do
```

The output looks like: 

```
# PARALLEL_PIXELS      =           2
# BITS_PER_CHANNEL     =          10
# CHANNELS             =           3
# BITS_PER_CHANNEL_AV  =          16
# BITS_PER_PIXEL_AV    =          48
# BITS_AV              =          96
# EMPTY_BITS           =           4
# BITS_PER_CHANNEL_AXI =          10
# BITS_PER_PIXEL_AXI   =          32
# BITS_AXI             =          64
# TUSER_BITS           =           8
# TUSER_FILL           =           6
# MASK_OUT             = 000003ff
# cin 0b0 | reset_n 0b0 | axm_tready 0bx | axm_tvalid 0bx | axm_tdata 0xXxxxxxxxXxxxxxxx | axm_tlast 0xx | axm_tuser 0x0X | asi_ready 0bx | asi_valid 0bx | asi_data 0xxxxxxxxxxxxxxxxxxxxxxxxx | asi_sop 0bx | asi_endofpacket 0bx | asi_empty 0bxxxx
# cin 0b1 | reset_n 0b0 | axm_tready 0bx | axm_tvalid 0bx | axm_tdata 0xXxxxxxxxXxxxxxxx | axm_tlast 0xx | axm_tuser 0x0X | asi_ready 0bx | asi_valid 0bx | asi_data 0xxxxxxxxxxxxxxxxxxxxxxxxx | asi_sop 0bx | asi_endofpacket 0bx | asi_empty 0bxxxx
# cin 0b0 | reset_n 0b1 | axm_tready 0bx | axm_tvalid 0bx | axm_tdata 0xXxxxxxxxXxxxxxxx | axm_tlast 0xx | axm_tuser 0x0X | asi_ready 0bx | asi_valid 0bx | asi_data 0xxxxxxxxxxxxxxxxxxxxxxxxx | asi_sop 0bx | asi_endofpacket 0bx | asi_empty 0bxxxx
# cin 0b1 | reset_n 0b1 | axm_tready 0bx | axm_tvalid 0bx | axm_tdata 0xXxxxxxxxXxxxxxxx | axm_tlast 0xx | axm_tuser 0x0X | asi_ready 0bx | asi_valid 0bx | asi_data 0xxxxxxxxxxxxxxxxxxxxxxxxx | asi_sop 0bx | asi_endofpacket 0bx | asi_empty 0bxxxx
# cin 0b0 | reset_n 0b1 | axm_tready 0b1 | axm_tvalid 0b1 | axm_tdata 0x0230882101304811 | axm_tlast 0x0 | axm_tuser 0x00 | asi_ready 0b1 | asi_valid 0b1 | asi_data 0x002300220021001300120011 | asi_sop 0b0 | asi_endofpacket 0b0 | asi_empty 0bxxxx
# cin 0b1 | reset_n 0b1 | axm_tready 0b1 | axm_tvalid 0b1 | axm_tdata 0x0230882101304811 | axm_tlast 0x0 | axm_tuser 0x00 | asi_ready 0b1 | asi_valid 0b1 | asi_data 0x002300220021001300120011 | asi_sop 0b0 | asi_endofpacket 0b0 | asi_empty 0bxxxx
# ** Note: $stop    : test_avalon_to_axi.sv(120)
#    Time: 30 ns  Iteration: 1  Instance: /test_avalon_to_axi
# Break at test_avalon_to_axi.sv line 120
# Stopped at test_avalon_to_axi.sv line 120
# End time: 07:29:43 on Apr 08,2024, Elapsed time: 0:00:00
# Errors: 0, Warnings: 0
```

```
vsim -c -do test_axi_to_avalon.do
```

The output should look like:

```
# PARALLEL_PIXELS      =           2
# BITS_PER_CHANNEL     =          10
# CHANNELS             =           3
# BITS_PER_CHANNEL_AV  =          16
# BITS_PER_PIXEL_AV    =          48
# BITS_AV              =          96
# EMPTY_BITS           =           4
# BITS_PER_CHANNEL_AXI =          10
# BITS_PER_PIXEL_AXI   =          32
# BITS_AXI             =          64
# TUSER_BITS           =           8
# TUSER_FILL           =           6
# MASK_OUT             = 000003ff
# cin 0b0 | reset_n 0b0 | axs_tready 0bx | axs_tvalid 0bx | axs_tdata 0xxxxxxxxxxxxxxxxx | axs_tlast 0xx | axs_tuser 0xxx | aso_ready 0bx | aso_valid 0bx | aso_data 0x0Xxx0Xxx0Xxx0Xxx0Xxx0Xxx | aso_sop 0bx | aso_endofpacket 0bx | aso_empty 0b0000
# cin 0b1 | reset_n 0b0 | axs_tready 0bx | axs_tvalid 0bx | axs_tdata 0xxxxxxxxxxxxxxxxx | axs_tlast 0xx | axs_tuser 0xxx | aso_ready 0bx | aso_valid 0bx | aso_data 0x0Xxx0Xxx0Xxx0Xxx0Xxx0Xxx | aso_sop 0bx | aso_endofpacket 0bx | aso_empty 0b0000
# cin 0b0 | reset_n 0b1 | axs_tready 0bx | axs_tvalid 0bx | axs_tdata 0xxxxxxxxxxxxxxxxx | axs_tlast 0xx | axs_tuser 0xxx | aso_ready 0bx | aso_valid 0bx | aso_data 0x0Xxx0Xxx0Xxx0Xxx0Xxx0Xxx | aso_sop 0bx | aso_endofpacket 0bx | aso_empty 0b0000
# cin 0b1 | reset_n 0b1 | axs_tready 0bx | axs_tvalid 0bx | axs_tdata 0xxxxxxxxxxxxxxxxx | axs_tlast 0xx | axs_tuser 0xxx | aso_ready 0bx | aso_valid 0bx | aso_data 0x0Xxx0Xxx0Xxx0Xxx0Xxx0Xxx | aso_sop 0bx | aso_endofpacket 0bx | aso_empty 0b0000
# cin 0b0 | reset_n 0b1 | axs_tready 0b1 | axs_tvalid 0b1 | axs_tdata 0xc2308821c1304811 | axs_tlast 0x0 | axs_tuser 0x00 | aso_ready 0b1 | aso_valid 0b1 | aso_data 0x002300220021001300120011 | aso_sop 0b0 | aso_endofpacket 0b0 | aso_empty 0b0000
# cin 0b1 | reset_n 0b1 | axs_tready 0b1 | axs_tvalid 0b1 | axs_tdata 0xc2308821c1304811 | axs_tlast 0x0 | axs_tuser 0x00 | aso_ready 0b1 | aso_valid 0b1 | aso_data 0x002300220021001300120011 | aso_sop 0b0 | aso_endofpacket 0b0 | aso_empty 0b0000
# ** Note: $stop    : test_axi_to_avalon.sv(120)
#    Time: 30 ns  Iteration: 1  Instance: /test_axi_to_avalon
# Break at test_axi_to_avalon.sv line 120
# Stopped at test_axi_to_avalon.sv line 120
# End time: 07:31:04 on Apr 08,2024, Elapsed time: 0:00:01
# Errors: 0, Warnings: 0
```