module oneapi_axi_to_avalon_gasket #(
    parameter int PARALLEL_PIXELS  = 1,
    parameter int BITS_PER_CHANNEL = 8,
    parameter int CHANNELS         = 3,

    parameter int BITS_PER_CHANNEL_AV  = 8,
    parameter int BITS_PER_PIXEL_AV    = 24,
    parameter int BITS_AV              = 24,
    parameter int EMPTY_BITS           = 2,
    parameter int BITS_PER_CHANNEL_AXI = 8,
    parameter int BITS_PER_PIXEL_AXI   = 24,
    parameter int BITS_AXI             = 24,
    parameter int TUSER_BITS           = 3,

    // one bit is used for endOfLine, one bit is used for interlaced. The rest
    // are filler.
    parameter int TUSER_FILL = 1,

    parameter int MASK_OUT = 'hff
) (
    // clk and resetn don't actually do anything; they are just there to keep
    // Platform Designer happy
    input logic csi_clk,
    input logic rsi_reset_n,

    // AXI4-Streaming Receiver Port
    output logic                  axs_tready,
    input  logic                  axs_tvalid,
    input  logic [  BITS_AXI-1:0] axs_tdata,
    input  logic                  axs_tlast,
    input  logic [TUSER_BITS-1:0] axs_tuser,

    // Avalon Streaming Source Port
    input  logic                  aso_ready,
    output logic                  aso_valid,
    output logic [   BITS_AV-1:0] aso_data,
    output logic                  aso_startofpacket,
    output logic                  aso_endofpacket,
    output logic [EMPTY_BITS-1:0] aso_empty
);

  // pick apart the output from VVP IP
  genvar px_id;
  genvar channel_id;
  generate

    for (px_id = 0; px_id < PARALLEL_PIXELS; px_id++) begin : parallel_pixel_assignment
      for (channel_id = 0; channel_id < CHANNELS; channel_id++) begin : parallel_channel_assignment
        logic [BITS_PER_CHANNEL_AV-1:0] pixel_chan;

        localparam LOWER_BIT_AXI = px_id * BITS_PER_PIXEL_AXI + channel_id * BITS_PER_CHANNEL_AXI;
        localparam UPPER_BIT_AXI = LOWER_BIT_AXI + BITS_PER_CHANNEL_AXI - 1;
        assign pixel_chan = axs_tdata[UPPER_BIT_AXI:LOWER_BIT_AXI] & MASK_OUT;

        localparam LOWER_BIT_AV = px_id * BITS_PER_PIXEL_AV + channel_id * BITS_PER_CHANNEL_AV;
        localparam UPPER_BIT_AV = LOWER_BIT_AV + BITS_PER_CHANNEL_AV - 1;
        assign aso_data[UPPER_BIT_AV:LOWER_BIT_AV] = pixel_chan;
      end
    end

  endgenerate

  // map to Avalon streaming port
  assign axs_tready        = aso_ready;
  assign aso_valid         = axs_tvalid;
  assign aso_startofpacket = axs_tuser[0];

  // eop will go high at the end of each line
  assign aso_endofpacket   = axs_tlast;

  // this signal violates Avalon, but don't worry about it since the oneAPI
  // kernel at the other end is designed to ignore `empty`. 
  assign aso_empty         = 0;
endmodule
