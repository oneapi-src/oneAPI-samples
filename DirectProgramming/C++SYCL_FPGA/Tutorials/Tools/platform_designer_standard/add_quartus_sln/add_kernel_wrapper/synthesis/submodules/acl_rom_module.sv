// (c) 1992-2024 Intel Corporation.                            
// Intel, the Intel logo, Intel, MegaCore, NIOS II, Quartus and TalkBack words    
// and logos are trademarks of Intel Corporation or its subsidiaries in the U.S.  
// and/or other countries. Other marks and brands may be claimed as the property  
// of others. See Trademarks on intel.com for full list of Intel trademarks or    
// the Trademarks & Brands Names Database (if Intel) or See www.Intel.com/legal (if Altera) 
// Your use of Intel Corporation's design tools, logic functions and other        
// software and tools, and its AMPP partner logic functions, and any output       
// files any of the foregoing (including device programming or simulation         
// files), and any associated documentation or information are expressly subject  
// to the terms and conditions of the Altera Program License Subscription         
// Agreement, Intel MegaCore Function License Agreement, or other applicable      
// license agreement, including, without limitation, that your use is for the     
// sole purpose of programming logic devices manufactured by Intel and sold by    
// Intel or its authorized distributors.  Please refer to the applicable          
// agreement for further details.                                                 


// The following are taken from the IP Generator (Qsys) generated instance of altera_syncram
// turn off superfluous verilog processor warnings
// altera message_level Level1
// altera message_off 10034 10035 10036 10037 10230 10240 10030

// acl_rom_module
//
// This module implements a pipelined ROM.  It accepts a read signal in, and several (depending on family) clock cycles later
// it outputs a readdatavalid signal and the correct data.  Address input only needs to be valid when read is asserted, and data
// output is only valid when readdatavalid is asserted.
//
// Reset can be synchronous or asynchronous (based on family).  For synchronous reset, the rst_n must be held asserted for
// a minimum of VALID_PIPE_DEPTH clock cycles (currently up to 5 cycles) to allow the readdatavalid pipeline to fully reset.
//
// Currently the ROM is forced into an M20K block

`default_nettype none

module acl_rom_module #(
    parameter ASYNC_RESET = 1,          // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter SYNCHRONIZE_RESET = 0,    // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
    parameter INIT_FILE = "sys_description.hex",
    parameter ADDRESS_WIDTH = 4,
    parameter DATA_WIDTH = 64,
    parameter FAMILY = "Arria 10",                   // "Cyclone V" | "Stratix V" | "Arria 10" | "Stratix 10", any unrecognized value is assumed to be a newer family and will be treated as Stratix 10
    parameter enable_ecc = "FALSE"                   // Enable error correction coding
) (
    input  wire                       clk,
    input  wire                       rst_n,            // reset is synchronous or asynchronous depending on ASYNC_RESET
                                                        // when synchronous, rst_n must be held asserted for a minimum of VALID_PIPE_DEPTH (see below) clock cycles
    input  wire  [ADDRESS_WIDTH-1:0]  address,          // word (not byte) based address, validated by the read signal
    input  wire                       read,             // when asserted, validates the address signal and causes valid data to be generate several (depending on VALID_PIPE_DEPTH) clock cycles later
    output logic [DATA_WIDTH-1:0]     readdata,         // output data, only valid when readdatavalid is asserted
    output logic                      readdatavalid,     // delayed version of read to match latency through the block, readdata is only guaranteed valid when this signal is asserted
    output logic [1:0]                ecc_err_status,    // ecc status signals
    output logic                      waitrequest        //no backpressure, tied to 0
);

    localparam     ADDR_PIPE_DEPTH      = ((FAMILY=="Cyclone V") || (FAMILY=="Stratix V") || (FAMILY=="Arria 10")) ? 0 : 1;             // pipeline the address for Stratix 10 and later families
    localparam     DATA_PIPE_DEPTH      = ((FAMILY=="Cyclone V") || (FAMILY=="Stratix V") || (FAMILY=="Arria 10")) ? 1 : 2;             // add an extra pipeline stage to the data path for Stratix 10 and later families
    localparam     VALID_PIPE_DEPTH     = ADDR_PIPE_DEPTH + DATA_PIPE_DEPTH + 2;                                                        // +2 for the delay through the altera_syncram

    logic                         readdatavalid_pipe      [1:VALID_PIPE_DEPTH];
    logic    [DATA_WIDTH-1:0]     readdata_pipe           [1:DATA_PIPE_DEPTH];
    logic                         sclrn;                                             // synchronous reset
    logic                         aclrn;                                             // asynchronous reset
    logic    [DATA_WIDTH-1:0]     readdata_ram;                                      // data output from the RAM block (registered out of the RAM megafunction)
    logic    [ADDRESS_WIDTH-1:0]  address_ram;                                       // address input to the RAM block (registered in the RAM megafunction)
    
    
    // connect either aclrn or sclrn to the input rst_n signal, tie the other off to a constant so it will get synthesized away
    acl_reset_handler
    #(
        .ASYNC_RESET            (ASYNC_RESET),
        .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
        .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
        .PIPE_DEPTH             (1),
        .NUM_COPIES             (1)
    )
    acl_reset_handler_inst
    (
        .clk                    (clk),
        .i_resetn               (rst_n),
        .o_aclrn                (aclrn),
        .o_resetn_synchronized  (),
        .o_sclrn                (sclrn)
    );
    
    // readdatavalid is control logic, must be reset
    always @(posedge clk or negedge aclrn) begin
        if (~aclrn) begin          // async reset will clear all bits in the readdatavalid pipeline
            for( int i=1; i<=VALID_PIPE_DEPTH; i++ ) begin : GEN_RANDOM_BLOCK_NAME_R60
                readdatavalid_pipe[i] <= 1'b0;
            end
        end
        else begin
            // simply shift the 'read' signal down the pipe
            readdatavalid_pipe[1] <= read;
            for( int i=2; i<=VALID_PIPE_DEPTH; i++ ) begin : GEN_RANDOM_BLOCK_NAME_R61
                readdatavalid_pipe[i] <= readdatavalid_pipe[i-1];
            end
            
            // synchronous reset (these assignments override the assignments above if sclrn is asserted)
            if (~sclrn) begin            // only clear the first and last bits in the readdatavalid pipeline, reset must be held asserted long enough for the first bit to propogate through the pipeline
                readdatavalid_pipe[1] <= 1'b0;
                readdatavalid_pipe[VALID_PIPE_DEPTH] <= 1'b0;
            end
        end
    end
    assign readdatavalid = readdatavalid_pipe[VALID_PIPE_DEPTH];

    // data path, no need for reset
    always @(posedge clk) begin
        readdata_pipe[1] <= readdata_ram;
        for( int i=2; i<=DATA_PIPE_DEPTH; i++ ) begin : GEN_RANDOM_BLOCK_NAME_R62
            readdata_pipe[i] <= readdata_pipe[i-1];
        end
    end
    assign readdata = readdata_pipe[DATA_PIPE_DEPTH];
    
    // address path, no need for reset, need to handle special case where pipeline depth = 0
    generate
        if (ADDR_PIPE_DEPTH==0) begin         // no pipeline, connect input address signals straight to the RAM
            assign address_ram = address;
        end
        else begin                            // implement a registered pipeline
            logic     [ADDRESS_WIDTH-1:0]  address_pipe            [1:ADDR_PIPE_DEPTH];
            always @(posedge clk) begin
                address_pipe[1] <= address;
                for( int i=2; i<=ADDR_PIPE_DEPTH; i++ ) begin : GEN_RANDOM_BLOCK_NAME_R63
                    address_pipe[i] <= address_pipe[i-1];
                end
            end
            assign address_ram = address_pipe[ADDR_PIPE_DEPTH];
        end
    endgenerate
    
    
    // instantiate the ROM
    acl_altera_syncram_wrapped #(
        .address_aclr_a( "NONE" ),
        .clock_enable_input_a( "BYPASS" ),
        .clock_enable_output_a( "BYPASS" ),
        .init_file( INIT_FILE ),
        .intended_device_family( FAMILY ),
        .lpm_hint( "ENABLE_RUNTIME_MOD=NO" ),
        .lpm_type( "altera_syncram" ),
        .operation_mode( "ROM" ),
        .outdata_aclr_a( "NONE" ),
        .outdata_sclr_a( "NONE" ),
        .outdata_reg_a( "CLOCK0" ),
        .ram_block_type( "M20K" ),             // Case:324281, set this back to "AUTO" when MLAB ROMs are functional with PR
        .widthad_a( ADDRESS_WIDTH ),
        .width_a( DATA_WIDTH ),
        .width_byteena_a( 1 ),
        .numwords_a( 2**ADDRESS_WIDTH ),
        .connect_clr_to_ram(0),
        .enable_ecc( "FALSE" ) // TODO: pass in enable_ecc after preencoding of the ROMs is done (Case:509038)
    ) altera_syncram_wrapped_component (
        .address_a (address_ram),
        .clock0 (clk),
        .q_a (readdata_ram),
        .aclr0 (~aclrn),
        .aclr1 (1'b0),
        .address_b (1'b1),
        .addressstall_a (1'b0),
        .addressstall_b (1'b0),
        .byteena_a (1'b1),
        .byteena_b (1'b1),
        .clock1 (1'b1),
        .clocken0 (1'b1),
        .clocken1 (1'b1),
        .data_a ({DATA_WIDTH{1'b1}}),
        .data_b (1'b1),
        //.eccencbypass (1'b0),
        //.eccencparity (8'b0),
        //.eccstatus ( ),
        .q_b ( ),
        .rden_a (1'b1),
        .rden_b (1'b1),
        .sclr (~sclrn),
        .wren_a (1'b0),
        .wren_b (1'b0),
        .ecc_err_status(ecc_err_status)
    );

   //tie off unused signals
   assign waitrequest = 1'b0;

endmodule

`default_nettype wire
