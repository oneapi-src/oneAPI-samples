//// (c) 1992-2024 Intel Corporation.                            
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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  ACL ECC DECODER
//
//  This module decodes data using a single error correct, double error detect Hamming code. As the data width get large,
//  so will the xor network and that would limit fmax. To resolve this, we slice the data into smaller groups and decode
//  each independently. Essentially we trade off more memory overhead for parity bits in order to limit the fmax
//  degradation due to ECC.
//
//  The user must specify the data width and the slicing size. From this, one can compute the number of parity bits and
//  total encoded bits (see the calculations in localparams below).
//
//  Error reporting: for each decoder (after slicing), there are 2 status signals: single error corrected, and double
//  error detected. Each of these signal types are OR-ed together from all of the decoders (from slicing) before being
//  reported to the outside world. Beware that if there are two bit errors but they are in separate slicing groups, two
//  independent decoders can correct one bit each, so this will be reported as single error corrected.
//
//  Reset: there is no reset. Pipeline stages are purely feed-forward, the intent is that reset will propagate through.
//
//  This module is actually a wrapper around the actual ECC implementation in secded_decoder. Here is the architecture.
//  For example, suppose DATA_WIDTH is 70 and ECC_GROUP_SIZE is 32, then we will slice input data into 32 + 32 + 6, and
//  3 encoders are used to produce 39 + 39 + 11 encoded bits.
//
//                                i_encoded[88:0]
//                                      |
//  +------------------------------------------------------------------------+
//  |                     optional input pipeline stages                     |
//  +------------------------------------------------------------------------+
//          |                           |                           |
//    encoded[88:78]              encoded[77:39]              encoded[38:0]
//          |                           |                           |
//  +----------------+          +----------------+          +----------------+
//  | secded_decoder |          | secded_decoder |          | secded_decoder |
//  +----------------+          +----------------+          +----------------+
//          |                           |                           |
//      data[69:64]                 data[63:32]                 data[31:0]
//          |                           |                           |
//  +------------------------------------------------------------------------+
//  |                     optional output pipeline stages                    |
//  +------------------------------------------------------------------------+
//                                      |
//                                o_data[69:0]
//
//  Everything decoder related is contained within this file. The related file that does the corresponding encoding is
//  acl_ecc_encoder.sv. Note both encoder and decoder require acl_ecc_pkg.sv.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

`default_nettype none

//BEWARE: do not leave the "clock_enable" input port disconnected if any pipeline stages are used, it will default to 0 and nothing will go through

module acl_ecc_decoder
import acl_ecc_pkg::*;
#(
    parameter int DATA_WIDTH,                   //number of bits in the decoded output data
    parameter int ECC_GROUP_SIZE,               //how many bits of unencoded data to group into one ecc block, see description in header comments
    parameter int INPUT_PIPELINE_STAGES = 0,    //number of pipeline stages between i_encoded and the ecc decoder
    parameter int OUTPUT_PIPELINE_STAGES = 0,   //number of pipeline stages between the ecc decoder and o_data
    parameter int STATUS_PIPELINE_STAGES = 0    //number of pipeline stages between the ecc decoder and o_single_error_corrected/o_double_error_detected
)
(
    input  wire                                                          clock,                     //clock is only needed if pipeline stages are nonzero
    input  wire                                                          clock_enable,              //set to 1 to sample i_encoded, intended for integration with altera_syncram, only needed if pipeline stages are nonzero
    input  wire  [getEncodedBitsEccGroup(DATA_WIDTH,ECC_GROUP_SIZE)-1:0] i_encoded,                 //encoded input data
    output logic [DATA_WIDTH-1:0]                                        o_data,                    //decoded output data
    output logic                                                         o_single_error_corrected,  //at least one ecc decoder corrected a single bit error within their ecc group
    output logic                                                         o_double_error_detected    //at least one ecc decoder detected a double bit error within their ecc group
);

    //helper functions for determining number of bits are defined in acl_ecc.svh
    localparam int ECC_NUM_GROUPS  = getNumGroups(DATA_WIDTH,ECC_GROUP_SIZE);           //how many groups to slice the data into
    localparam int LAST_GROUP_SIZE = getLastGroupSize(DATA_WIDTH,ECC_GROUP_SIZE);       //all groups have size ECC_GROUP_SIZE except possibly the last group which may be smaller since it gets the remaining bits
    localparam int ENCODED_BITS    = getEncodedBitsEccGroup(DATA_WIDTH,ECC_GROUP_SIZE);

    //internal signals
    genvar g;
    logic [ENCODED_BITS-1:0] encoded;
    logic [DATA_WIDTH-1:0] data;
    logic [2*ECC_NUM_GROUPS-1:0] error_status;
    logic [ECC_NUM_GROUPS-1:0] single_error_corrected;
    logic [ECC_NUM_GROUPS-1:0] double_error_detected;

    //input pipeline stages
    generate
    if (INPUT_PIPELINE_STAGES == 0) begin
        assign encoded = i_encoded;
    end
    else begin
        logic [ENCODED_BITS-1:0] encoded_pipe [INPUT_PIPELINE_STAGES-1:0];
        always_ff @(posedge clock) begin    //only the first pipeline stage needs a clock enable, the remaining pipeline stages will load the same data when the clock enable propagates there
            if (clock_enable) encoded_pipe[0] <= i_encoded;
        end
        for (g=1; g<INPUT_PIPELINE_STAGES; g++) begin : gen_input_pipe
            always_ff @(posedge clock) begin
                encoded_pipe[g] <= encoded_pipe[g-1];
            end
        end
        assign encoded = encoded_pipe[INPUT_PIPELINE_STAGES-1];
    end
    endgenerate

    //slice the data for each decoder
    generate
    for (g=0; g<ECC_NUM_GROUPS; g++) begin : gen_decoder
        localparam int RAW_BASE = ECC_GROUP_SIZE*g;
        localparam int ENC_BASE = getEncodedBits(ECC_GROUP_SIZE)*g;
        localparam int RAW_WIDTH = (g==ECC_NUM_GROUPS-1) ? LAST_GROUP_SIZE : ECC_GROUP_SIZE;
        localparam int ENC_WIDTH = getEncodedBits(RAW_WIDTH);

        secded_decoder #(
            .DATA_WIDTH               (RAW_WIDTH)
        )
        secded_encoder_inst
        (
            .i_encoded                (encoded[ENC_BASE +: ENC_WIDTH]),
            .o_data                   (data[RAW_BASE +: RAW_WIDTH]),
            .o_single_error_corrected (error_status[g]),
            .o_double_error_detected  (error_status[g+ECC_NUM_GROUPS])
        );
    end
    endgenerate

    //output pipeline stages
    generate
    if (OUTPUT_PIPELINE_STAGES == 0) begin
        assign o_data = data;
    end
    else begin
        logic [DATA_WIDTH-1:0] data_pipe [OUTPUT_PIPELINE_STAGES-1:0];
        if (INPUT_PIPELINE_STAGES == 0) begin    //this is the first pipeline stage
            always_ff @(posedge clock) begin
                if (clock_enable) data_pipe[0] <= data;
            end
        end
        else begin  //there was a previous pipeline in the input stage which would have captured the clock enable
            always_ff @(posedge clock) begin
                data_pipe[0] <= data;
            end
        end
        for (g=1; g<OUTPUT_PIPELINE_STAGES; g++) begin : gen_output_pipe
            always_ff @(posedge clock) begin
                data_pipe[g] <= data_pipe[g-1];
            end
        end
        assign o_data = data_pipe[OUTPUT_PIPELINE_STAGES-1];
    end
    endgenerate

    //error status pipeline stages
    generate
    if (STATUS_PIPELINE_STAGES == 0) begin
        assign {double_error_detected, single_error_corrected} = error_status;
    end
    else begin
        logic [2*ECC_NUM_GROUPS-1:0] error_status_pipe [STATUS_PIPELINE_STAGES-1:0];
        if (INPUT_PIPELINE_STAGES == 0) begin    //this is the first pipeline stage
            always_ff @(posedge clock) begin
                if (clock_enable) error_status_pipe[0] <= error_status;
            end
        end
        else begin  //there was a previous pipeline in the input stage which would have captured the clock enable
            always_ff @(posedge clock) begin
                error_status_pipe[0] <= error_status;
            end
        end
        for (g=1; g<STATUS_PIPELINE_STAGES; g++) begin : gen_status_pipe
            always_ff @(posedge clock) begin
                error_status_pipe[g] <= error_status_pipe[g-1];
            end
        end
        assign {double_error_detected, single_error_corrected} = error_status_pipe[STATUS_PIPELINE_STAGES-1];
    end
    endgenerate
    assign o_single_error_corrected = |single_error_corrected;
    assign o_double_error_detected = |double_error_detected;

endmodule
//end acl_ecc_decoder





// Hamming code decoder, single error correct, double error detect
//
// This implementation follows the bit mapping as shown on Wikipedia, parity bits are added at power of 2 locations, data bits go in between
// For example, with DATA_WIDTH = 11, we have 4 Hamming parity bits and one overall parity bit, so the bit locations will looks like this, d means data, p means parity
// [0] = p0, [1] = p1, [2] = p2, [3] = d0, [4] = p3, [5] = d1, [6] = d2, [7] = d3, [8] = p4, [9] = d4, [10] = d5, [11] = d6, [12] = d7, [13] = d8, [14] = d9, [15] = d10

module secded_decoder
import acl_ecc_pkg::*;
#(
    parameter int DATA_WIDTH
) (
    input  wire  [getEncodedBits(DATA_WIDTH)-1:0] i_encoded,                //encoded input data
    output logic [DATA_WIDTH-1:0]                 o_data,                   //decoded output data
    output logic                                  o_single_error_corrected, //asserts when one bit of encoded data is wrong, this will be reported and corrected
    output logic                                  o_double_error_detected   //asserts when two bits of encoded data are wrong, this will only be reported and not corrected
);

    //helper functions for determining number of bits are defined in acl_ecc.svh
    localparam int PARITY_BITS = getParityBits(DATA_WIDTH);
    localparam int ENCODED_BITS = getEncodedBits(DATA_WIDTH);

    //compute the parity bits
    logic [PARITY_BITS-1:0] parity;
    always_comb begin
        for (int parity_index=1; parity_index<PARITY_BITS; parity_index++) begin : GEN_RANDOM_BLOCK_NAME_R7
            parity[parity_index] = 0;
            for (int enc_index=0; enc_index<ENCODED_BITS; enc_index++) begin : GEN_RANDOM_BLOCK_NAME_R8
                if (enc_index & (1<<(parity_index-1))) begin   //bit parity_index-1 of enc_index is 1
                    parity[parity_index] = parity[parity_index] ^ i_encoded[enc_index]; //running xor
                end
            end
        end
        parity[0] = ^i_encoded; //overall parity
    end

    //syndrome indicates which bits was wrong, if any
    logic [PARITY_BITS-2:0] syndrome;
    assign syndrome = parity[PARITY_BITS-1:1];

    //report if there was 1 bit or 2 bit errors respectively
    assign o_single_error_corrected = parity[0];    //odd number of errors, 1 error gets corrected, 3 errors is not correctable and mapping to the word of minimum hamming distance will give incorrect data
    assign o_double_error_detected = ~parity[0] && (syndrome != 0);    //even number of errors, 0 errors results in syndrome == 0, 2 error will have a nonzero syndrome

    //extract out the data bits, and correct if there is a single bit error
    //parity bits are at power of 2 bit locations, data bits are in between
    //for example, with DATA_WIDTH = 11, we have 5 parity bits and the bit locations will looks like this, d means data, p means parity
    //[0] = p0, [1] = p1, [2] = p2, [3] = d0, [4] = p3, [5] = d1, [6] = d2, [7] = d3, [8] = p4, [9] = d4, [10] = d5, [11] = d6, [12] = d7, [13] = d8, [14] = d9, [15] = d10
    always_comb begin
        for (int enc_index=0, data_index=0; enc_index<ENCODED_BITS; enc_index++) begin : GEN_RANDOM_BLOCK_NAME_R9
            if (!(enc_index == 0 || (2**$clog2(enc_index)) == enc_index)) begin    //enc_index is not a power of 2
                o_data[data_index] = (enc_index==syndrome) ? ~i_encoded[enc_index] : i_encoded[enc_index];
                data_index++;
            end
        end
    end

endmodule
//end secded_decoder

`default_nettype wire
