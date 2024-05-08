module test_axi_to_avalon;

  `define PARALLEL_PIXELS 4
  `define BITS_PER_CHANNEL 12
  `define CHANNELS 3

  `define BITS_PER_CHANNEL_AV (1 << $clog2(`BITS_PER_CHANNEL))
  `define BITS_PER_PIXEL_AV (`BITS_PER_CHANNEL_AV * `CHANNELS)
  `define BITS_AV (`BITS_PER_PIXEL_AV * `PARALLEL_PIXELS)
  `define EMPTY_BITS ($clog2(`BITS_AV / 8))
  `define BITS_PER_CHANNEL_AXI (`BITS_PER_CHANNEL)
  `define BITS_PER_PIXEL_AXI (8 * ((`CHANNELS * `BITS_PER_CHANNEL_AXI + 7) / 8))
  `define BITS_AXI (`BITS_PER_PIXEL_AXI * `PARALLEL_PIXELS)
  `define TUSER_BITS ((`BITS_AXI + 7)/ 8)
  `define TUSER_FILL (`TUSER_BITS - 2)
  `define MASK_OUT ((1 << `BITS_PER_CHANNEL_AXI) - 1)


  // clk and resetn don't actually do anything; they are just there to keep
  // Platform Designer happy
  logic                   cin;
  logic                   reset_n;

  // AXI4 Receiver
  logic                   axs_tready;
  logic                   axs_tvalid;
  logic [  `BITS_AXI-1:0] axs_tdata;
  logic                   axs_tlast;
  logic [`TUSER_BITS-1:0] axs_tuser;

  // Avalon Streaming Source Port
  logic                   aso_ready;
  logic                   aso_valid;
  logic [   `BITS_AV-1:0] aso_data;
  logic                   aso_startofpacket;
  logic                   aso_endofpacket;
  logic [`EMPTY_BITS-1:0] aso_empty;

  // Instantiate the Device Under Test (DUT)
  oneapi_axs_to_avs_pixel_gasket #(
        `PARALLEL_PIXELS     ,
        `BITS_PER_CHANNEL    ,
        `CHANNELS            ,
        `BITS_PER_CHANNEL_AV ,
        `BITS_PER_PIXEL_AV   ,
        `BITS_AV             ,
        `EMPTY_BITS          ,
        `BITS_PER_CHANNEL_AXI,
        `BITS_PER_PIXEL_AXI  ,
        `BITS_AXI            ,
        `TUSER_BITS          ,
        `TUSER_FILL          ,
        `MASK_OUT ) 
  axi_to_avalon (
      .csi_clk(cin),
      .rsi_reset_n(reset_n),

      // AXI4-Streaming Receiver Port
      .axs_tready(axs_tready),
      .axs_tvalid(axs_tvalid),
      .axs_tdata (axs_tdata),
      .axs_tlast (axs_tlast),
      .axs_tuser (axs_tuser),

      // Avalon Streaming Source Port
      .aso_ready(aso_ready),
      .aso_valid(aso_valid),
      .aso_data(aso_data),
      .aso_startofpacket(aso_startofpacket),
      .aso_endofpacket(aso_endofpacket),
      .aso_empty(aso_empty)
  );

  initial begin
    cin     = 0;
    reset_n = 0;
    while (1) begin
      #5;
      cin = ~cin;
    end
  end

  // generate data for axs test vector
  for (genvar px_id = 0; px_id < `PARALLEL_PIXELS; px_id++) begin : parallel_pixel_assignment
    for (genvar channel_id = 0; channel_id < `CHANNELS; channel_id++) begin : parallel_channel_assignment
      logic [`BITS_PER_CHANNEL_AXI-1:0] pixel_chan;

      assign pixel_chan = (16 * (px_id + 1) + (channel_id + 1)) & `MASK_OUT;

      localparam LOWER_BIT_AXI = px_id * `BITS_PER_PIXEL_AXI + channel_id * `BITS_PER_CHANNEL_AXI;
      localparam UPPER_BIT_AXI = LOWER_BIT_AXI + `BITS_PER_CHANNEL_AXI - 1;
      assign axs_tdata[UPPER_BIT_AXI:LOWER_BIT_AXI] = pixel_chan;
    end
  end

  // padding bits for axs  
  for (genvar px_id2 = 0; px_id2 < `PARALLEL_PIXELS; px_id2++) begin : padding_assignment
    localparam PADDING_BITS_AXI  = `BITS_PER_PIXEL_AXI - (`BITS_PER_CHANNEL_AXI * `CHANNELS);
    localparam LOWER_PADDING_AXI = px_id2 * `BITS_PER_PIXEL_AXI + `CHANNELS * `BITS_PER_CHANNEL_AXI;
    localparam UPPER_PADDING_AXI = LOWER_PADDING_AXI + PADDING_BITS_AXI - 1;
    if (PADDING_BITS_AXI !== 0) begin
      assign axs_tdata[UPPER_PADDING_AXI:LOWER_PADDING_AXI] = 0;
    end
  end

  initial begin
    $display("PARALLEL_PIXELS      = %d", `PARALLEL_PIXELS);
    $display("BITS_PER_CHANNEL     = %d", `BITS_PER_CHANNEL);
    $display("CHANNELS             = %d", `CHANNELS);
    $display("BITS_PER_CHANNEL_AV  = %d", `BITS_PER_CHANNEL_AV);
    $display("BITS_PER_PIXEL_AV    = %d", `BITS_PER_PIXEL_AV);
    $display("BITS_AV              = %d", `BITS_AV);
    $display("EMPTY_BITS           = %d", `EMPTY_BITS);
    $display("BITS_PER_CHANNEL_AXI = %d", `BITS_PER_CHANNEL_AXI);
    $display("BITS_PER_PIXEL_AXI   = %d", `BITS_PER_PIXEL_AXI);
    $display("BITS_AXI             = %d", `BITS_AXI);
    $display("TUSER_BITS           = %d", `TUSER_BITS);
    $display("TUSER_FILL           = %d", `TUSER_FILL);
    $display("MASK_OUT             = %x", `MASK_OUT);
  end

  logic [`BITS_PER_PIXEL_AXI-1:0] pixel1_axs;
  logic [`BITS_PER_PIXEL_AXI-1:0] pixel2_axs;

  initial begin
    @(negedge cin);
    reset_n = 1'b1;
    @(negedge cin);
    axs_tvalid <= 1;
    axs_tlast  <= 0;
    axs_tuser  <= 0;

    aso_ready  <= 1;
    @(negedge cin);

    $display("---------------------------------------------------------------------------------");
    $display(
        "cin 0b%b | reset_n 0b%b | axs_tready 0b%b | axs_tvalid 0b%b | axs_tdata 0x%h | axs_tlast 0x%h | axs_tuser 0x%h ",
        cin, reset_n, axs_tready, axs_tvalid, axs_tdata, axs_tlast, axs_tuser);
    $display(
        "cin 0b%b | reset_n 0b%b | aso_ready  0b%b | aso_valid  0b%b | aso_data  0x%h | aso_sop   0b%b | aso_endofpacket 0b%b | aso_empty 0b%b",
        cin, reset_n, aso_ready, aso_valid, aso_data, aso_startofpacket, aso_endofpacket, aso_empty);
    $stop;
  end

endmodule
