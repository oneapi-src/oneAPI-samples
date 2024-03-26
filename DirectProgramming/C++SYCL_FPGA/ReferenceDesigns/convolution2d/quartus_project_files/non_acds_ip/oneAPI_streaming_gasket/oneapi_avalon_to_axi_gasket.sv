module oneapi_avalon_to_axi_gasket #(
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

    // Avalon Streaming Sink Port
    output logic                  asi_ready,
    input  logic                  asi_valid,
    input  logic [   BITS_AV-1:0] asi_data,
    input  logic                  asi_startofpacket,
    input  logic                  asi_endofpacket,
    input  logic [EMPTY_BITS-1:0] asi_empty,

    // AXI4-Streaming Transmitter Port
    input  logic                  axm_tready,
    output logic                  axm_tvalid,
    output logic [  BITS_AXI-1:0] axm_tdata,
    output logic                  axm_tlast,
    output logic [TUSER_BITS-1:0] axm_tuser
);

  // pick apart the output from oneAPI IP
  genvar px_id;
  genvar channel_id;
  generate

    for (px_id = 0; px_id < PARALLEL_PIXELS; px_id++) begin : parallel_pixel_assignment
      for (channel_id = 0; channel_id < CHANNELS; channel_id++) begin : parallel_channel_assignment
        logic [BITS_PER_CHANNEL_AXI-1:0] pixel_chan;

        localparam LOWER_BIT_AV = px_id * BITS_PER_PIXEL_AV + channel_id * BITS_PER_CHANNEL_AV;
        localparam UPPER_BIT_AV = LOWER_BIT_AV + BITS_PER_CHANNEL_AXI - 1;
        assign pixel_chan = asi_data[UPPER_BIT_AV:LOWER_BIT_AV];

        localparam LOWER_BIT_AXI = px_id * BITS_PER_PIXEL_AXI + channel_id * BITS_PER_CHANNEL_AXI;
        localparam UPPER_BIT_AXI = LOWER_BIT_AXI + BITS_PER_CHANNEL_AXI - 1;
        assign axm_tdata[UPPER_BIT_AXI:LOWER_BIT_AXI] = pixel_chan;
      end
    end
  endgenerate


  // padding bits for axi
  localparam PADDING_BITS_AXI = BITS_PER_PIXEL_AXI - (BITS_PER_CHANNEL_AXI * CHANNELS);
  genvar px_id2;
  generate
    for (px_id2 = 0; px_id2 < PARALLEL_PIXELS; px_id2++) begin : padding_assignment

      localparam LOWER_PADDING_AXI = px_id2 * BITS_PER_PIXEL_AXI + CHANNELS * BITS_PER_CHANNEL_AXI;
      localparam UPPER_PADDING_AXI = LOWER_PADDING_AXI + PADDING_BITS_AXI - 1;
      if (PADDING_BITS_AXI !== 0) begin
        assign axm_tdata[UPPER_PADDING_AXI:LOWER_PADDING_AXI] = 0;
      end
    end
  endgenerate

  // map to AXI4-S port
  assign asi_ready  = axm_tready;
  assign axm_tvalid = asi_valid;

  // don't support interlaced video, so ignore TUSER[1].
  logic [TUSER_FILL-1:0] tuser_fill = 0;
  assign axm_tuser = {tuser_fill, 1'b0, asi_startofpacket};

  // eop will go high at the end of each line
  assign axm_tlast = asi_endofpacket;

  // ignore empty
endmodule
