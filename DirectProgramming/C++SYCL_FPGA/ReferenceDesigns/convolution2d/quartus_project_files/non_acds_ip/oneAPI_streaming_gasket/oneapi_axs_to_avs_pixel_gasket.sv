//  Allow an Intel Streaming Video AXI4 Streaming interface to connect to a
//  oneAPI IP

module oneapi_axs_to_avs_pixel_gasket #(
    // The number of pixels that are processed in a single streaming
    // transaction.
    parameter int PARALLEL_PIXELS  = 1,

    // The number of bits in each color channel on the AXI4-S interface. Padding
    // bits will be added to each channel of the Avalon Streaming interface to
    // adapt to limitations in the oneAPI compiler.
    parameter int BITS_PER_CHANNEL = 8,

    // The number of color channels in each pixel. For example, RGB pixels would
    // have 3 channels.
    parameter int CHANNELS         = 3,

    // The following derived parameters are calculated in the associated _hw.tcl
    // file. They control how the pixels are mapped into the asi_data signal of
    // the Avalon streaming interface, and the axm_tdata signal of the AXI
    // streaming interface.

    // Avalon streaming parameters
    parameter int BITS_PER_CHANNEL_AV  = 8,
    parameter int BITS_PER_PIXEL_AV    = 24,
    parameter int BITS_AV              = 24,
    parameter int EMPTY_BITS           = 2,

    // AXI streaming parameters
    parameter int BITS_PER_CHANNEL_AXI = 8,
    parameter int BITS_PER_PIXEL_AXI   = 24,
    parameter int BITS_AXI             = 24,
    parameter int TUSER_BITS           = 3,

    // one bit is used for endOfLine, one bit is used for interlaced. The rest
    // are filler.
    parameter int TUSER_FILL           = 1,
    
    // mask unused bits in the AXI transmitter port
    parameter int MASK_OUT             = 'hff
) (
    // clk and resetn don't actually do anything; they are just there to keep
    // Platform Designer happy
    input logic csi_clk,
    input logic rsi_reset_n,

    // AXI4-Streaming Receiver Port
    output logic                  axs_tready,
    input  logic                  axs_tvalid,
    // see Readme.md for how pixels map to the tdata signal
    input  logic [  BITS_AXI-1:0] axs_tdata,
    input  logic                  axs_tlast,

    // TUSER[0] indicates start of frame
    // the rest of TUSER is unused.
    input  logic [TUSER_BITS-1:0] axs_tuser,

    // Avalon Streaming Source Port
    input  logic                  aso_ready,
    output logic                  aso_valid,
    // see Readme.md for how pixels map to the data signal
    output logic [   BITS_AV-1:0] aso_data,
    output logic                  aso_startofpacket,
    output logic                  aso_endofpacket,
    output logic [EMPTY_BITS-1:0] aso_empty
);

  // Pick apart the output from oneAPI IP. Remap pixels and insert padding as
  // shown in Readme.md
  for (genvar px_id = 0; px_id < PARALLEL_PIXELS; px_id++) begin : parallel_pixel_assignment
    for (genvar channel_id = 0; channel_id < CHANNELS; channel_id++) begin : parallel_channel_assignment
      logic [BITS_PER_CHANNEL_AV-1:0] pixel_chan;

      localparam LOWER_BIT_AXI = px_id * BITS_PER_PIXEL_AXI + channel_id * BITS_PER_CHANNEL_AXI;
      localparam UPPER_BIT_AXI = LOWER_BIT_AXI + BITS_PER_CHANNEL_AXI - 1;
      assign pixel_chan = axs_tdata[UPPER_BIT_AXI:LOWER_BIT_AXI] & MASK_OUT;

      localparam LOWER_BIT_AV = px_id * BITS_PER_PIXEL_AV + channel_id * BITS_PER_CHANNEL_AV;
      localparam UPPER_BIT_AV = LOWER_BIT_AV + BITS_PER_CHANNEL_AV - 1;
      assign aso_data[UPPER_BIT_AV:LOWER_BIT_AV] = pixel_chan;
    end
  end

  assign axs_tready        = aso_ready;
  assign aso_valid         = axs_tvalid;

  // tuser[0] will go high at the start of a new frame
  assign aso_startofpacket = axs_tuser[0];

  // tlast will go high at the end of each line
  assign aso_endofpacket   = axs_tlast;

  // this signal violates Avalon, but don't worry about it since the oneAPI
  // kernel at the other end is designed to ignore `empty`. 
  assign aso_empty         = 0;
endmodule
