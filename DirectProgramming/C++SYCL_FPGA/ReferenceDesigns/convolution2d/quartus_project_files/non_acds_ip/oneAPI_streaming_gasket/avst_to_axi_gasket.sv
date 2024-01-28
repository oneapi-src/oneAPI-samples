module avalon_to_axi_gasket(

    // clk and resetn don't actually do anything; they are just there to keep
    // Platform Designer happy
    input logic clk,
    input logic resetn,

    // Avalon Streaming Sink Port
    output logic        avst_sink_ready,
    input  logic        avst_sink_valid,
    input  logic [95:0] avst_sink_data,
    input  logic        avst_sink_sop,
    input  logic        avst_sink_eop,
    input  logic [3:0]  avst_sink_empty,

    // AXI4-Streaming Transmitter Port
    input  logic        axi_tx_tready,
    output logic        axi_tx_tvalid,
    output logic [63:0] axi_tx_tdata,
    output logic        axi_tx_tlast,
    output logic [7:0]  axi_tx_tuser
);

    // pick apart the output from oneAPI IP
    logic [9:0] pixel0_r;
    logic [9:0] pixel0_g;
    logic [9:0] pixel0_b;

    logic [9:0] pixel1_r;
    logic [9:0] pixel1_g;
    logic [9:0] pixel1_b;

    assign pixel0_r = avst_sink_data[41:32];
    assign pixel0_g = avst_sink_data[25:16];
    assign pixel0_b = avst_sink_data[9:0];

    assign pixel1_r = avst_sink_data[89:80];
    assign pixel1_g = avst_sink_data[73:64];
    assign pixel1_b = avst_sink_data[57:48];


    // map to AXI4-S port
    assign avst_sink_ready = axi_tx_tready;
    assign axi_tx_tvalid   = avst_sink_valid;
    assign axi_tx_tdata    = {2'b0, pixel1_r, pixel1_g, pixel1_b, 2'b0, pixel0_r, pixel0_g, pixel0_b};

    // don't support interlaced video, so ignore TUSER[1].
    assign axi_tx_tuser    = {6'b0, 1'b0, avst_sink_sop};

    // eop will go high at the end of each line
    assign axi_tx_tlast    = avst_sink_eop;

    // ignore empty
endmodule


