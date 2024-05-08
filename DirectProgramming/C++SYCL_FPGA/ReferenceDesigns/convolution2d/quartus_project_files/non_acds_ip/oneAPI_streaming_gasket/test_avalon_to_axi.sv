module test_avalon_to_axi;

  `define PARALLEL_PIXELS 4
  `define BITS_PER_CHANNEL 10
  `define CHANNELS 4

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

  // Avalon Streaming Sink Port
  logic                   asi_ready;
  logic                   asi_valid;
  logic [   `BITS_AV-1:0] asi_data;
  logic                   asi_startofpacket;
  logic                   asi_endofpacket;
  logic [`EMPTY_BITS-1:0] asi_empty;

  // AXI4 Transmitter
  logic                   axm_tready;
  logic                   axm_tvalid;
  logic [  `BITS_AXI-1:0] axm_tdata;
  logic                   axm_tlast;
  logic [`TUSER_BITS-1:0] axm_tuser;

  // Instantiate the Device Under Test (DUT)
  oneapi_avs_to_axs_pixel_gasket #(
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
  avalon_to_axi (
      .csi_clk(cin),
      .rsi_reset_n(reset_n),

      // Avalon Streaming Sink Port
      .asi_ready(asi_ready),
      .asi_valid(asi_valid),
      .asi_data(asi_data),
      .asi_startofpacket(asi_startofpacket),
      .asi_endofpacket(asi_endofpacket),
      .asi_empty(asi_empty),

      // AXI4-Streaming Transmitter Port
      .axm_tready(axm_tready),
      .axm_tvalid(axm_tvalid),
      .axm_tdata (axm_tdata),
      .axm_tlast (axm_tlast),
      .axm_tuser (axm_tuser)
  );

  initial begin
    cin     = 0;
    reset_n = 0;
    while (1) begin
      #5;
      cin = ~cin;
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

  // generate data for asi test vector
  generate
    for (genvar px_id = 0; px_id < `PARALLEL_PIXELS; px_id++) begin : parallel_pixel_assignment
      for (genvar channel_id = 0; channel_id < `CHANNELS; channel_id++) begin : parallel_channel_assignment
        localparam LOWER_BIT_AV = px_id * `BITS_PER_PIXEL_AV + channel_id * `BITS_PER_CHANNEL_AV;
        localparam UPPER_BIT_AV = LOWER_BIT_AV + `BITS_PER_CHANNEL_AV - 1;
        
        logic [`BITS_PER_CHANNEL_AV-1:0] pixel_chan;
        assign pixel_chan = (16 * (px_id + 1) + (channel_id + 1)) & `MASK_OUT;

        assign asi_data[UPPER_BIT_AV:LOWER_BIT_AV] = pixel_chan;
      end
    end
  endgenerate

  initial begin
    @(negedge cin);
    reset_n = 1'b1;
    @(negedge cin);
    asi_valid <= 1;
    asi_endofpacket  <= 0;
    asi_startofpacket  <= 0;

    axm_tready  <= 1;

    @(negedge cin);

    $display("---------------------------------------------------------------------------------");
    $display(
        "cin 0b%b | reset_n 0b%b | axm_tready 0b%b | axm_tvalid 0b%b | axm_tdata 0x%h | axm_tlast 0x%h | axm_tuser       0x%h ",
        cin, reset_n, axm_tready, axm_tvalid, axm_tdata, axm_tlast, axm_tuser);
    $display(
        "cin 0b%b | reset_n 0b%b | asi_ready  0b%b | asi_valid  0b%b | asi_data  0x%h | asi_sop   0b%b | asi_endofpacket 0b%b | asi_empty 0b%b",
        cin, reset_n, asi_ready, asi_valid, asi_data, asi_startofpacket, asi_endofpacket, asi_empty);
    $stop;
  end

endmodule
