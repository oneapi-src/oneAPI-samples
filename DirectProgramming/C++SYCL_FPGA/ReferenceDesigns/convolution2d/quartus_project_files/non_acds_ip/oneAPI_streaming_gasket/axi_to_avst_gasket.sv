module axi_to_avalon_gasket(

    // clk and resetn don't actually do anything; they are just there to keep
    // Platform Designer happy
    input logic clk,
    input logic resetn,

    // AXI4-Streaming Recevier Port
    output logic        axi_rx_tready,
    input  logic        axi_rx_tvalid,
    input  logic [63:0] axi_rx_tdata,
    input  logic        axi_rx_tlast,
    input  logic [7:0]  axi_rx_tuser,

    // Avalon Streaming Source Port
    input  logic        avst_source_ready,
    output logic        avst_source_valid,
    output logic [95:0] avst_source_data,
    output logic        avst_source_sop,
    output logic        avst_source_eop,
    output logic [3:0]  avst_source_empty
);

    // pick apart the output from VVP IP
    logic [9:0] pixel0_r;
    logic [9:0] pixel0_g;
    logic [9:0] pixel0_b;

    logic [9:0] pixel1_r;
    logic [9:0] pixel1_g;
    logic [9:0] pixel1_b;

    assign pixel0_r = axi_rx_tdata[29:20];
    assign pixel0_g = axi_rx_tdata[19:10];
    assign pixel0_b = axi_rx_tdata[9:0];

    assign pixel1_r = axi_rx_tdata[61:52];
    assign pixel1_g = axi_rx_tdata[51:42];
    assign pixel1_b = axi_rx_tdata[41:32];

    // map to Avalon streaming port
    assign axi_rx_tready     = avst_source_ready;
    assign avst_source_valid = axi_rx_tvalid;
    assign avst_source_data  = {6'b0, pixel1_r, 6'b0, pixel1_g, 6'b0, pixel1_b, 6'b0, pixel0_r, 6'b0, pixel0_g, 6'b0, pixel0_b};
    assign avst_source_sop   = axi_rx_tuser[0];
    
    // eop will go high at the end of each line
    assign avst_source_eop   = axi_rx_tlast; 

    // this signal violates Avalon, but don't worry about it since the oneAPI
    // kernel at the other end is designed to ignore `empty`. 
    assign avst_source_empty = 0;

endmodule


