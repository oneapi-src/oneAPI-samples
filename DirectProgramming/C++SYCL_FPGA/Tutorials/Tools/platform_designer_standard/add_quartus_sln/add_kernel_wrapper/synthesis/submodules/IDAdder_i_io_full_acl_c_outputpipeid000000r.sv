// ------------------------------------------------------------------------- 
// High Level Design Compiler for Intel(R) FPGAs Version 2024.2.1 (Release Build #a1b6f61c6e)
// 
// Legal Notice: Copyright 2024 Intel Corporation.  All rights reserved.
// Your use of  Intel Corporation's design tools,  logic functions and other
// software and  tools, and its AMPP partner logic functions, and any output
// files any  of the foregoing (including  device programming  or simulation
// files), and  any associated  documentation  or information  are expressly
// subject  to the terms and  conditions of the  Intel FPGA Software License
// Agreement, Intel MegaCore Function License Agreement, or other applicable
// license agreement,  including,  without limitation,  that your use is for
// the  sole  purpose of  programming  logic devices  manufactured by  Intel
// and  sold by Intel  or its authorized  distributors. Please refer  to the
// applicable agreement for further details.
// ---------------------------------------------------------------------------

// SystemVerilog created from i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_0gr
// Created for function/kernel IDAdder
// SystemVerilog created on Tue Aug 20 10:13:47 2024


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name AUTO_ROM_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module IDAdder_i_io_full_acl_c_outputpipeid000000r (
    input wire [0:0] in_avst_iowr_nb_acl_c_OutputPipeID_pipe_channel_almostfull,
    input wire [0:0] in_i_stall,
    output wire [0:0] out_o_almostfull,
    output wire [0:0] out_o_valid,
    output wire [0:0] out_o_stall,
    input wire [0:0] in_i_valid,
    input wire clock,
    input wire resetn
    );

    wire [0:0] i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr_out_o_data;
    wire [0:0] i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr_out_o_stall;
    wire [0:0] i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr_out_o_valid;
    wire [0:0] i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_in;
    wire i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_in_bitsignaltemp;
    wire [0:0] i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_out;
    wire i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_out_bitsignaltemp;


    // i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr(EXTIFACE,3)
    assign i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_in = in_avst_iowr_nb_acl_c_OutputPipeID_pipe_channel_almostfull;
    assign i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_in_bitsignaltemp = i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_in[0];
    assign i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_out[0] = i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_out_bitsignaltemp;
    acl_dspba_buffer #(
        .WIDTH(1)
    ) thei_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr (
        .buffer_in(i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_in_bitsignaltemp),
        .buffer_out(i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_out_bitsignaltemp)
    );

    // i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr(BLACKBOX,2)@20000000
    IDAdder_i_io_full_acl_c_outputpipeid000001r thei_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr (
        .in_i_data(i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_52_1gr_buffer_out),
        .in_i_stall(in_i_stall),
        .in_i_valid(in_i_valid),
        .out_o_data(i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr_out_o_data),
        .out_o_stall(i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr_out_o_stall),
        .out_o_valid(i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr_out_o_valid),
        .clock(clock),
        .resetn(resetn)
    );

    // sync_out_10(GPOUT,6)@1
    assign out_o_almostfull = i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr_out_o_data;
    assign out_o_valid = i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr_out_o_valid;

    // sync_out_9(GPOUT,7)@20000000
    assign out_o_stall = i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr_out_o_stall;

endmodule
