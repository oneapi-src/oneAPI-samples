// Allow a oneAPI IP to connect to an Intel Streaming Video AXI4 Streaming
// interface. This IP should synthesize into wires.

module oneapi_avs_to_axs_pixel_gasket #(
    // The value of this parameter is the number of pixels that are processed in
    // a single streaming transaction.
    parameter int PARALLEL_PIXELS  = 1,

    // The number of bits in each color channel on the AXI4-S interface. Padding
    // bits will be added to each channel of the Avalon Streaming interface to
    // adapt to limitations in the oneAPI compiler.
    parameter int BITS_PER_CHANNEL = 8,

    // The number of color channels in each pixel. For example, an RGB pixel
    // would have 3 channels. 
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

    // Avalon Streaming Sink Port
    output logic                  asi_ready,
    input  logic                  asi_valid,
    // see Readme.md for how pixels map to the data signal
    input  logic [   BITS_AV-1:0] asi_data,
    input  logic                  asi_startofpacket,
    input  logic                  asi_endofpacket,
    input  logic [EMPTY_BITS-1:0] asi_empty,

    // AXI-Streaming Transmitter Port
    input  logic                  axm_tready,
    output logic                  axm_tvalid,
    // see Readme.md for how pixels map to the tdata signal
    output logic [  BITS_AXI-1:0] axm_tdata,
    output logic                  axm_tlast,

    // TUSER[0] indicates start of frame
    // the rest of TUSER is unused.
    output logic [TUSER_BITS-1:0] axm_tuser
);

  // Pick apart the output from oneAPI IP. Remap pixels and insert padding as
  // shown in Readme.md
  for (genvar px_id = 0; px_id < PARALLEL_PIXELS; px_id++) begin : parallel_pixel_assignment
    for (genvar channel_id = 0; channel_id < CHANNELS; channel_id++) begin : parallel_channel_assignment
      logic [BITS_PER_CHANNEL_AXI-1:0] pixel_chan;

      localparam LOWER_BIT_AV = px_id * BITS_PER_PIXEL_AV + channel_id * BITS_PER_CHANNEL_AV;
      localparam UPPER_BIT_AV = LOWER_BIT_AV + BITS_PER_CHANNEL_AXI - 1;
      assign pixel_chan = asi_data[UPPER_BIT_AV:LOWER_BIT_AV];

      localparam LOWER_BIT_AXI = px_id * BITS_PER_PIXEL_AXI + channel_id * BITS_PER_CHANNEL_AXI;
      localparam UPPER_BIT_AXI = LOWER_BIT_AXI + BITS_PER_CHANNEL_AXI - 1;
      assign axm_tdata[UPPER_BIT_AXI:LOWER_BIT_AXI] = pixel_chan;
    end
  end

  // padding bits for axi  
  for (genvar px_id2 = 0; px_id2 < PARALLEL_PIXELS; px_id2++) begin : padding_assignment
    localparam PADDING_BITS_AXI  = BITS_PER_PIXEL_AXI - (BITS_PER_CHANNEL_AXI * CHANNELS);
    localparam LOWER_PADDING_AXI = px_id2 * BITS_PER_PIXEL_AXI + CHANNELS * BITS_PER_CHANNEL_AXI;
    localparam UPPER_PADDING_AXI = LOWER_PADDING_AXI + PADDING_BITS_AXI - 1;
    if (PADDING_BITS_AXI !== 0) begin
      assign axm_tdata[UPPER_PADDING_AXI:LOWER_PADDING_AXI] = 0;
    end
  end

  assign asi_ready  = axm_tready;
  assign axm_tvalid = asi_valid;

  logic [TUSER_FILL-1:0] tuser_fill = 0;

  // startofpacket will go high at the start of a new frame
  assign axm_tuser = {tuser_fill, 1'b0, asi_startofpacket};

  // endofpacket will go high at the end of each line
  assign axm_tlast = asi_endofpacket;

  // ignore empty because the oneAPI kernel does not produce `empty`.
endmodule
