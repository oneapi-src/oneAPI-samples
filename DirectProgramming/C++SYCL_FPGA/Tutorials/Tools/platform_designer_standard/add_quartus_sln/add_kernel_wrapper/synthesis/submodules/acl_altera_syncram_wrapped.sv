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


///////////////////////////////////////////////////////////////////////////////////////
//                                                                                   //
// ACL ALTERA SYNCRAM WRAPPED                                                        //
//                                                                                   //
// DESCRIPTION                                                                       //
// ===========                                                                       //
// This is a wrapper around the altera_syncram component and behaves in almost       //
// the exact same way when the parameter enable_ecc is set to "FALSE". The           //
// module has the added (soft) ECC feature.                                          //
//                                                                                   //
// The module has the exact same interface except for                                //
//   * Two addtional parameters:                                                     //
//     - "enable_ecc": accepts "TRUE" or "FALSE". The default is "FALSE"             //
//     - "connect_clr_to_ram": accepts 1'b0 or 1'b1 and controls whether the aclr    //
//       and the sclr signals need to be connected to the altera_syncram component   //
//       or only to the encoder/decoder.                                             //
//   * One additional output port:                                                   //
//     - "ecc_err_status": this is a 2-bit status signal that behaves as follows:    //
//       ----------------------------------------------------------------            //
//       Status       Meaning                                                        //
//       ----------------------------------------------------------------            //
//       00           No error detected                                              //
//       10           Error detected and corrected (single bit error)                //
//       x1           Error detected but uncorrectable (double bit error)            //
//       ----------------------------------------------------------------            //
//                                                                                   //
// Required files                                                                    //
// ==============                                                                    //
//    acl_ecc_encoder.sv                                                             //
//    acl_ecc_decoder.sv                                                             //
//    acl_ecc_pkg.sv                                                                 //
//                                                                                   //
// Usage                                                                             //
// =====                                                                             //
// Only if using A10, this module can be used with the "enable_ecc" parameter        //
// set to "TRUE" to enable ECC. When ECC is disabled, the wrapper can be used        //
// with any family in almost the exact same way as the altera_syncram IP.            //
//                                                                                   //
// Notes                                                                             //
// =====                                                                             //
// Notable differences between this and altera_syncram:                              //
// - The S10 additional ports: address2_a, address2_b, eccencparity,                 //
//   eccencbypass, are not available here.                                           //
// - The S10 additional paremter width_eccencparity is not available here.           //
// - The two ports clocken2 and clocken3 are not avilable here because they are      //
//   not supported on S10 and they cause clear box issues. Plus, we don't use        //
//   them anywhere.                                                                  //
// - This was taken from the simulation model for altera_syncram which was           //
//   using tri0 or tri1 for the input/output ports.  As it is not recommended to     //
//   use tri0/tri1, we changed all inputs ports to be of type  "wire" and all        //
//   output ports to be of type "logic". In simulation, everything seems to be       //
//   okay with this change. All the testbenches pass as expected. Will keep an       //
//   eye on this in the hardware runs to see if it's okay.                           //
// - On Stratix 10 and in BIDIR_DUAL_PORT, it is not allowed to use                  //
//   addressstall_a and addressstall_b, so we do not connect those for the           //
//   BIDIR_DUAL_PORT case for all families.                                          //
//                                                                                   //
// - Byteenables:                                                                    //
//   * If ECC is off, then we decide whether to connect the byte enable ports        //
//     depending on the byte enable width specified. If the width is 1, then we      //
//     don't connect since the defaul is 1, which means the parameter is not         //
//     manually specified.                                                           //
//   * If ECC is on, then we decide wether to connect the byte enable ports          //
//     depending on the new parameter use_byteena which is set to 1 in the           //
//     backend if byte enables are truly used.  This parameter is set to 1 if the    //
//     store instruction is byteenabled OR if the width of the data to store is      //
//     different than the width of the memory. This is only observed for             //
//     acl_mem1x and acl_mem2x. So, when ECC is ON, ECC slice size is set to 8,      //
//     and after adding the parity bits and padding with zeros, the resulting        //
//     memory width will double, so will the width of the byte enalbe signals        //
//                                                                                   //
///////////////////////////////////////////////////////////////////////////////////////

`default_nettype none

module acl_altera_syncram_wrapped
import acl_ecc_pkg::*;
#(

  /*------------------------------------\
  |  Original altera_syncram paramters  |
  \------------------------------------*/

  // PORT A PARAMETERS
  parameter width_a          = 1,
  parameter widthad_a        = 1,
  parameter widthad2_a       = 1,
  parameter numwords_a       = 0,
  parameter outdata_reg_a    = "UNREGISTERED",
  parameter address_aclr_a   = "NONE",
  parameter outdata_aclr_a   = "NONE",
  parameter width_byteena_a  = 1,

  // PORT B PARAMETERS
  parameter width_b                   = 1,
  parameter widthad_b                 = 1,
  parameter widthad2_b                = 1,
  parameter numwords_b                = 0,
  parameter rdcontrol_reg_b           = "CLOCK1",
  parameter address_reg_b             = "CLOCK1",
  parameter outdata_reg_b             = "UNREGISTERED",
  parameter outdata_aclr_b            = "NONE",
  parameter indata_reg_b              = "CLOCK1",
  parameter byteena_reg_b             = "CLOCK1",
  parameter address_aclr_b            = "NONE",
  parameter width_byteena_b           = 1,

  // Clock Enable Parameters
  parameter clock_enable_input_a  = "NORMAL",         // Valid possible values: NORMAL, BYPASS, ALTERNATE
  parameter clock_enable_output_a = "NORMAL",         // Valid possible values: NORMAL, BYPASS
  parameter clock_enable_input_b  = "NORMAL",         // Valid possible values: NORMAL, BYPASS, ALTERNATE
  parameter clock_enable_output_b = "NORMAL",         // Valid possible values: NORMAL, BYPASS
  parameter clock_enable_core_a   = "USE_INPUT_CLKEN",  // Valid possible values: USE_INPUT_CLKEN, NORMAL, BYPASS, ALTERNATE
  parameter clock_enable_core_b   = "USE_INPUT_CLKEN",  // Valid possible values: USE_INPUT_CLKEN, NORMAL, BYPASS, ALTERNATE

  // Read During Write Paramters
  parameter read_during_write_mode_port_a      = "NEW_DATA_NO_NBE_READ",
  parameter read_during_write_mode_port_b      = "NEW_DATA_NO_NBE_READ",
  parameter read_during_write_mode_mixed_ports = "DONT_CARE",

   // NADDER NEW FEATURES
  parameter outdata_sclr_a            = "NONE",
  parameter outdata_sclr_b            = "NONE",
  parameter enable_coherent_read      = "FALSE",
  parameter enable_force_to_zero      = "FALSE",

  // GLOBAL PARAMETERS
  parameter operation_mode            = "BIDIR_DUAL_PORT",
  parameter byte_size                 = 0,
  parameter ram_block_type            = "AUTO",
  parameter init_file                 = "UNUSED",
  parameter init_file_layout          = "UNUSED",
  parameter maximum_depth             = 0,
  parameter intended_device_family    = "Arria 10",
  parameter lpm_hint                  = "UNUSED",
  parameter lpm_type                  = "altsyncram",
  parameter implement_in_les          = "OFF",
  parameter power_up_uninitialized    = "FALSE",

  /*----------------\
  |  New parameters  |
  \----------------*/

  // ECC RELATED PARAMETERS
  parameter enable_ecc                = "FALSE", // in the future, this may turn on hard ecc or soft ecc, depending on the RAM type, for now only soft ecc is supported
  parameter connect_clr_to_ram        = 0,       // If 1, the pass in the aclr and sclr signal to RAM ,aclr and scrl will now also be used for register from ecc decoder signals

  parameter do_not_connect_addressstall = 0,  // Prevent connecting addressstall when unused because this prevents Quartus from putting AUTO into an MLAB
  parameter do_not_connect_read_enable  = 0,  // If 1, then do not connect read enable ports


  // BYTE ENABLE SUPPORT
  parameter use_byteena               = 0        // If 1, then byte enables are truly used and we must handle that case separately when ECC is ON.
)
(

  /*------------------------------------\
  |  Original altera_syncram I/O ports  |
  \------------------------------------*/

  // INPUT PORT DECLARATION
  input wire                 wren_a,    // Port A write/read enable input
  input wire                 wren_b,    // Port B write enable input
  input wire                 rden_a,    // Port A read enable input
  input wire                 rden_b,    // Port B read enable input
  input wire   [width_a-1:0] data_a,    // Port A data input
  input wire   [width_b-1:0] data_b,    // Port B data input
  input wire [widthad_a-1:0] address_a, // Port A address input
  input wire [widthad_b-1:0] address_b, // Port B address input

  // clock inputs on both ports and here are their usage
  // Port A -- 1. all input registers must be clocked by clock0.
  //           2. output register can be clocked by either clock0, clock1 or none.
  // Port B -- 1. all input registered must be clocked by either clock0 or clock1.
  //           2. output register can be clocked by either clock0, clock1 or none.

  input wire  clock0,
  input wire  clock1,

  // clock enable inputs and here are their usage
  // clocken0 -- can only be used for enabling clock0.
  // clocken1 -- can only be used for enabling clock1.

  input wire clocken0,
  input wire clocken1,

  // clear inputs on both ports and here are their usage
  // Port A -- 1. all input registers can only be cleared by clear0 or none.
  //           2. output register can be cleared by either clear0, clear1 or none.
  // Port B -- 1. all input registers can be cleared by clear0, clear1 or none.
  //           2. output register can be cleared by either clear0, clear1 or none.

  input wire aclr0,
  input wire aclr1,

  input wire [width_byteena_a-1:0] byteena_a, // Port A byte enable input
  input wire [width_byteena_b-1:0] byteena_b, // Port B byte enable input

  // Stratix II related ports
  input wire addressstall_a,
  input wire addressstall_b,

  // Nadder new features - Stratix 10 onwards
  input wire sclr,

  // OUTPUT PORT DECLARATION
  output logic [width_a-1:0] q_a, // Port A output
  output logic [width_b-1:0] q_b, // Port B output


  /*----------------\
  |  New I/O ports  |
  \----------------*/

  output logic         [1:0] ecc_err_status  //10 = error detected and corrected (memory in not updated), 01 = error detected but uncorrectable
);

  // Check if the two port widths match in the case where width_b is set by the user (that is != 1). We do not support different port widths.
  // Also check if the enable_ecc parameter is legal.
  // The checks are done in Quartus pro and Modelsim, they are disabled in Quartus standard because they results in a syntax error (parser is based on an older systemverilog standard)
  // the workaround is to use synthesis translate to hide this from Quartus standard, ALTERA_RESERVED_QHD is only defined in Quartus pro, and Modelsim ignores the synthesis comment
  `ifdef ALTERA_RESERVED_QHD
  `else
  //synthesis translate_off
  `endif
  generate
    if ((width_b != 1) && (width_a != width_b)) begin
      $fatal(1, "acl_altera_syncram_wrapped: Values of width do not match. width_a:%d width_b:%d \n", width_a, width_b);
    end
    if ((enable_ecc != "TRUE") && (enable_ecc != "FALSE")) begin
      $fatal(1, "acl_altera_syncram_wrapped: illegal value of enable_ecc = %s, legal values are \"TRUE\" or \"FALSE\"\n", enable_ecc);
    end
  endgenerate
  `ifdef ALTERA_RESERVED_QHD
  `else
  //synthesis translate_on
  `endif

  // Calculate the internal memory width (width of the codewords)
  // Even though we compute both memory_width_a and memory_width_b, we do not currently support different port widths.
  // We make the distinction here to avoid errors in the single port case.
  localparam int MAX_ECC_WIDTH    = (use_byteena) ? 8 : 32;  //this value should be ECC_GROUP_SIZE and can be swept to tradeoff fmax vs memory overhead. Do not set this value larger than 64 due to altecc implementation
  localparam int NUM_GROUPS       = getNumGroups(width_a, MAX_ECC_WIDTH);
  localparam int MEMORY_WIDTH_A   = (enable_ecc == "TRUE") ? getEncodedBitsEccGroup(width_a, MAX_ECC_WIDTH) : width_a; // width of codewords coming out of the encoder at port a
  localparam int MEMORY_WIDTH_B   = (enable_ecc == "TRUE") ? getEncodedBitsEccGroup(width_b, MAX_ECC_WIDTH) : width_b; // width of codewords coming out of the encoder at port b

  // BYTE ENABLE SUPPORT
  localparam int I_MEMORY_WIDTH_A   = (enable_ecc == "TRUE") ? ((use_byteena == 1) ? MAX_ECC_WIDTH*2*NUM_GROUPS : MEMORY_WIDTH_A) : width_a;  // width of the codewords in the RAM at port a.
  localparam int I_MEMORY_WIDTH_B   = (enable_ecc == "TRUE") ? ((use_byteena == 1) ? MAX_ECC_WIDTH*2*NUM_GROUPS : MEMORY_WIDTH_B) : width_b;  // width of the codewords in the RAM at port b.
  localparam int I_WIDTH_BYTEENA_A  = (enable_ecc == "TRUE" && use_byteena == 1) ? 2*width_byteena_a : width_byteena_a;   // width of the byteenable signal for port a. That doubles if ECC is used and use_byteena = 1.
  localparam int I_WIDTH_BYTEENA_B  = (enable_ecc == "TRUE" && use_byteena == 1) ? 2*width_byteena_b : width_byteena_b;   // width of the byteenable signal for port b. That doubles if ECC is used and use_byteena = 1.

  localparam int ERR_CORRECTED_PULSE_EXTENSION_WIDTH = 3;  // amount of pulse stretching for the "error corrected" signal.

  // signals for making internal connections
  //
  // Data flow diagram:
  //         ---------                  ------------                    ----------------                    -------------                  ---------
  // data--->|encoder|---codeword_wr--->|pad groups|---i_codeword_wr--->|altera_syncram|---i_codeword_rd--->|trim groups|---codeword_rd--->|decoder|--->q
  //         ---------                  ------------                    ----------------                    -------------                  ---------
  // If ECC is OFF: the encoder, the decoder, the padding, and the trimming blocks are pass-through
  // If ECC is ON and use_byteena = 1, all the blocks are there.
  // If ECC is ON and use_byteena = 0, the padding and the trimming blocks are pass-through.


  logic     [MEMORY_WIDTH_A-1:0] codeword_wr_a;
  logic     [MEMORY_WIDTH_A-1:0] codeword_rd_a;

  logic     [MEMORY_WIDTH_B-1:0] codeword_wr_b;
  logic     [MEMORY_WIDTH_B-1:0] codeword_rd_b;

  logic   [I_MEMORY_WIDTH_A-1:0] i_codeword_wr_a;
  logic   [I_MEMORY_WIDTH_A-1:0] i_codeword_rd_a;

  logic   [I_MEMORY_WIDTH_B-1:0] i_codeword_wr_b;
  logic   [I_MEMORY_WIDTH_B-1:0] i_codeword_rd_b;

  logic [I_WIDTH_BYTEENA_A-1: 0] i_byteena_a;     // Internal byteenable signal for port a, which has double the width if ECC is on
  logic [I_WIDTH_BYTEENA_B-1: 0] i_byteena_b;     // Internal byteenable signal for port b, which has double the width if ECC is on



  logic aclr0_int;                                 // aclr0 signal to be sent to altera_syncram
  logic aclr1_int;                                 // aclr1 signal to be sent to altera_syncram
  logic sclr_int;                                  // sclr signal to be sent to altera_syncram

  logic err_fatal;                                 // ecc status signal for detected uncorrected error
  logic err_corrected;                             // ecc status signal for detected corrected error

  ////////////////////////////////////////////////////////
  //INTERNAL WIRE DECLARATIONS

  wire i_outdata_clken_a;
  wire i_outdata_clken_b;
  wire i_clock1;
  wire i_clocken1;
  wire i_clocken0;
  wire i_clocken1_b;
  wire i_clocken0_b;
  wire i_in_data_clken_b;

  wire i_ecc_enc_clk_a;
  wire i_ecc_dec_clk_a;
  wire i_ecc_enc_clk_b;
  wire i_ecc_dec_clk_b;

  wire i_ecc_aclr;

  // SIGNAL ASSIGNMENT
  // Clock signal assignment  for ecc encoder and decoder
  assign i_ecc_enc_clk_a                = clock0; // Port A input register is always clocked with clock0
  assign i_ecc_dec_clk_a                = (outdata_reg_a == "CLOCK0") ?
                                          clock0 : (outdata_reg_a == "CLOCK1") ?
                                          clock1 :  i_ecc_enc_clk_a; // If output is unregistered, use the input clock.

  // Clear box doesn't understand it if clock1 is set to 1'b1 when the wrapper is instantiated.
  // Therefore, we need to use heuristics here to manually disconnect to drive some of the ports, based on the
  // combination of parameters.
  // This needs to be done for clock1 and clocken1
  // *****************************************
  // legal operations for all operation modes:
  //      |  PORT A  |  PORT B  |
  //      |  RD  WR  |  RD  WR  |
  // BDP  |  x   x   |  x   x   |
  // DP   |      x   |  x       |
  // SP   |  x   x   |          |
  // ROM  |  x       |          |
  // *****************************************

  // Note that the clock for indata_reg_b should be equal to the clock address_reg_b
  assign i_clock1   = (operation_mode == "ROM")
                      || ((operation_mode == "SINGLE_PORT") &&  (outdata_reg_a !="CLOCK1"))
                      || ((operation_mode == "DUAL_PORT") && (outdata_reg_b != "CLOCK1") && (address_reg_b != "CLOCK1"))
                      || ((operation_mode == "BIDIR_DUAL_PORT") && (outdata_reg_a != "CLOCK1") && (outdata_reg_b != "CLOCK1" ) && (address_reg_b != "CLOCK1")) ?
                      1'b1 : clock1;
  assign i_clocken1 = (operation_mode == "ROM")
                      || ((operation_mode == "SINGLE_PORT") &&  (outdata_reg_a !="CLOCK1"))
                      || ((operation_mode == "DUAL_PORT") && (outdata_reg_b != "CLOCK1") && (address_reg_b != "CLOCK1"))
                      || ((operation_mode == "BIDIR_DUAL_PORT") && (outdata_reg_a != "CLOCK1") && (outdata_reg_b != "CLOCK1" ) && (address_reg_b != "CLOCK1")) ?
                      1'b1 : clocken1;

  // Note that clock for indata_reg_b should be equal to the clock for address_reg_b
  assign i_ecc_enc_clk_b                = (address_reg_b == "CLOCK0") ?
                                          clock0 : clock1; // Port A input register is always clocked with clock0
  assign i_ecc_dec_clk_b                = (outdata_reg_b == "CLOCK0") ?
                                          clock0 : (outdata_reg_b == "CLOCK1") ?
                                          clock1 :  i_ecc_enc_clk_b; // If output is unregistered, use the input clock.

  // Clock enable signal assignment
  // port a clock enable assignments:
  assign i_outdata_clken_a              = (clock_enable_output_a == "BYPASS") ?
                                          1'b1 : (outdata_reg_a == "CLOCK1") ?
                                          clocken1 : (outdata_reg_a == "CLOCK0") ?
                                          clocken0 : 1'b1;
  // port b clock enable assignments:
  assign i_outdata_clken_b              = (clock_enable_output_b == "BYPASS") ?
                                          1'b1 : (outdata_reg_b == "CLOCK1") ?
                                          clocken1 : (outdata_reg_b == "CLOCK0") ?
                                          clocken0 : 1'b1;

  assign i_ecc_aclr                     = aclr0 | aclr1;   // The pulse extender and the latch for both ports should reset on any clear.

  assign i_clocken0                     = (clock_enable_input_a == "BYPASS") ?
                                          1'b1 : clocken0;

  assign i_clocken0_b                   = (clock_enable_input_b == "BYPASS") ?
                                          1'b1 : clocken0;

  assign i_clocken1_b                   = (clock_enable_input_b == "BYPASS") ?
                                          1'b1 : clocken1;

  assign i_in_data_clken_b              = (address_reg_b == "CLOCK0") ? i_clocken0_b : i_clocken1_b;

  // aclr and sclr assignments.
  generate
    if( connect_clr_to_ram != 1) begin
      assign aclr0_int = 1'b0;
      assign aclr1_int = 1'b0;
      assign sclr_int  = 1'b0;
    end else begin
      assign aclr0_int = aclr0;
      assign aclr1_int = aclr1;
      assign sclr_int  = sclr;
  end
  endgenerate

  genvar i;
  generate
    if (enable_ecc == "TRUE") begin                           : GEN_ECC_ENABLED

      logic err_corrected_int_a;
      logic err_corrected_pulse_extended_a;

      logic err_corrected_int_b;
      logic err_corrected_pulse_extended_b;

      logic err_fatal_int_a;
      logic err_fatal_latched_a;

      logic err_fatal_int_b;
      logic err_fatal_latched_b;

      logic [ERR_CORRECTED_PULSE_EXTENSION_WIDTH-1:0] d_a;
      logic [ERR_CORRECTED_PULSE_EXTENSION_WIDTH-1:0] d_b;

      // instantiate the ECC encoder for port a
      acl_ecc_encoder #(
         .DATA_WIDTH                    (width_a),
         .ECC_GROUP_SIZE                (MAX_ECC_WIDTH),
         .INPUT_PIPELINE_STAGES         (0),
         .OUTPUT_PIPELINE_STAGES        (0)
      ) acl_ecc_encoder_inst_a (
         .clock                         (i_ecc_enc_clk_a),
         .clock_enable                  (i_clocken0),        // This will be added after ecc_enc/dec update
         .i_data                        (data_a),
         .o_encoded                     (codeword_wr_a)
      );

      // instantiate the ECC encoder for port b
      acl_ecc_encoder #(
         .DATA_WIDTH                    (width_b),
         .ECC_GROUP_SIZE                (MAX_ECC_WIDTH),
         .INPUT_PIPELINE_STAGES         (0),
         .OUTPUT_PIPELINE_STAGES        (0)
      ) acl_ecc_encoder_inst_b (
         .clock                         (i_ecc_enc_clk_b),
         .clock_enable                  (i_in_data_clken_b),   // This will be added after ecc_enc/dec update
         .i_data                        (data_b),
         .o_encoded                     (codeword_wr_b)
      );

      if (use_byteena) begin
        genvar g;
        for (g = 0; g < NUM_GROUPS; g++) begin : pad_codewords
           localparam int new_enc_base = g*MAX_ECC_WIDTH*2;
           localparam int old_enc_base = g*getEncodedBits(MAX_ECC_WIDTH);
           localparam int enc_width    = getEncodedBits(MAX_ECC_WIDTH);

           assign i_codeword_wr_a[new_enc_base +: enc_width] = codeword_wr_a[old_enc_base +: enc_width];
           assign i_codeword_wr_a[new_enc_base+enc_width +: 2*MAX_ECC_WIDTH-enc_width] = '0;

           assign i_codeword_wr_b[new_enc_base +: enc_width] = codeword_wr_b[old_enc_base +: enc_width];
           assign i_codeword_wr_b[new_enc_base+enc_width +: 2*MAX_ECC_WIDTH-enc_width] = '0;

           assign codeword_rd_a[old_enc_base +: enc_width] = i_codeword_rd_a[new_enc_base +: enc_width];

           assign codeword_rd_b[old_enc_base +: enc_width] = i_codeword_rd_b[new_enc_base +: enc_width];
        end

        for (g = 0; g < width_byteena_a; g++) begin : adjust_byteena_a
           assign i_byteena_a[2*g]   = byteena_a[g];
           assign i_byteena_a[2*g+1] = byteena_a[g];
        end

        for (g = 0; g < width_byteena_b; g++) begin : adjust_byteena_b
           assign i_byteena_b[2*g]   = byteena_b[g];
           assign i_byteena_b[2*g+1] = byteena_b[g];
        end
      end
      else begin
        assign i_codeword_wr_a = codeword_wr_a;
        assign i_codeword_wr_b = codeword_wr_b;
        assign   codeword_rd_a = i_codeword_rd_a;
        assign   codeword_rd_b = i_codeword_rd_b;
        assign i_byteena_a     = byteena_a;
        assign i_byteena_b     = byteena_b;
      end

      // instantiate the ECC decoder for port a
      acl_ecc_decoder #(
         .DATA_WIDTH                    (width_a),
         .ECC_GROUP_SIZE                (MAX_ECC_WIDTH),
         .INPUT_PIPELINE_STAGES         (0),
         .OUTPUT_PIPELINE_STAGES        (0),
         .STATUS_PIPELINE_STAGES        (0)
      ) acl_ecc_decoder_inst_a (
         .clock                         (i_ecc_dec_clk_a),
         .clock_enable                  (i_outdata_clken_a), // This will be added after ecc_enc/dec update
         .i_encoded                     (codeword_rd_a),
         .o_single_error_corrected      (err_corrected_int_a),      //Flag signal to reflect the status of data received. Denotes single-bit error found and corrected. You can use the data because it has already been corrected.
         .o_double_error_detected       (err_fatal_int_a),          // Flag signal to reflect the status of data received. Denotes double-bit error found, but not corrected. You must not use the data if this signal is asserted.
         .o_data                        (q_a)
      );

      // instantiate the ECC decoder for port b
      acl_ecc_decoder #(
         .DATA_WIDTH                    (width_b),
         .ECC_GROUP_SIZE                (MAX_ECC_WIDTH),
         .INPUT_PIPELINE_STAGES         (0),
         .OUTPUT_PIPELINE_STAGES        (0),
         .STATUS_PIPELINE_STAGES        (0)
      ) acl_ecc_decoder_inst_b (
         .clock                         (i_ecc_dec_clk_b),
         .clock_enable                  (i_outdata_clken_b), // This will be added after ecc_enc/dec update
         .i_encoded                     (codeword_rd_b),
         .o_single_error_corrected      (err_corrected_int_b),      //Flag signal to reflect the status of data received. Denotes single-bit error found and corrected. You can use the data because it has already been corrected.
         .o_double_error_detected       (err_fatal_int_b),          // Flag signal to reflect the status of data received. Denotes double-bit error found, but not corrected. You must not use the data if this signal is asserted.
         .o_data                        (q_b)
      );

      always_ff @(posedge i_ecc_dec_clk_a or posedge i_ecc_aclr) begin
         if (i_ecc_aclr) begin
            err_fatal_latched_a            <= 1'b0;
            err_corrected_pulse_extended_a <= 1'b0;
            d_a                            <=  'b0;
         end
         else begin
            if (rden_a) begin
               err_fatal_latched_a <= err_fatal_latched_a | err_fatal_int_a;

               d_a[0] <= err_corrected_int_a;
               for (int i = 1; i < ERR_CORRECTED_PULSE_EXTENSION_WIDTH; i++) begin : pulse_extend_a
                  d_a[i] <= d_a[i-1];
               end

               err_corrected_pulse_extended_a <= ( | d_a) | err_corrected_int_a;
            end

            if (sclr) begin
               err_fatal_latched_a            <= 1'b0;
               err_corrected_pulse_extended_a <= 1'b0;
               d_a                            <=  'b0;
            end
         end
      end

      always_ff @(posedge i_ecc_dec_clk_b or posedge i_ecc_aclr) begin
         if (i_ecc_aclr) begin
            err_fatal_latched_b            <= 1'b0;
            err_corrected_pulse_extended_b <= 1'b0;
            d_b                            <=  'b0;

         end
         else begin
            if (rden_b) begin
               err_fatal_latched_b <= err_fatal_latched_b | err_fatal_int_b;
               d_b[0] <= err_corrected_int_b;
               for (int i = 1; i < ERR_CORRECTED_PULSE_EXTENSION_WIDTH; i++) begin : pulse_extend_b
                  d_b[i] <= d_b[i-1];
               end

               err_corrected_pulse_extended_b <= ( | d_b) | err_corrected_int_b;
            end

            if (sclr) begin
               err_fatal_latched_b            <= 1'b0;
               err_corrected_pulse_extended_b <= 1'b0;
               d_b                            <=  'b0;
            end
         end
      end


      // Note that the latched and pulse extended registers from port A and port B may generally be clocked differently.
      // We are oring them here asyncronously, since from here, resulting or is async. This will not cause a problem as long as
      // both of the ports use the same clock (currently always the case in OpenCL), or the status goes into a syncronizer.

      assign err_fatal      = err_fatal_latched_a            | err_fatal_latched_b;
      assign err_corrected  = err_corrected_pulse_extended_a | err_corrected_pulse_extended_b;
      assign ecc_err_status = {err_corrected,err_fatal};

    end else begin                                  : GEN_ECC_DISABLED   // soft_ecc is false

       assign err_corrected    = 1'b0;
       assign err_fatal        = 1'b0;
       assign ecc_err_status   = {err_corrected,err_fatal};
       assign i_codeword_wr_a  = data_a;
       assign i_codeword_wr_b  = data_b;
       assign q_a              = i_codeword_rd_a;
       assign q_b              = i_codeword_rd_b;
       assign i_byteena_a      = byteena_a;
       assign i_byteena_b      = byteena_b;

    end

  endgenerate

  /*----------------------------------------\
  |  Internal altera_syncram instantiation  |
  \----------------------------------------*/

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                                                                                        //
  // If ECC is ON                                                                                                                           //
  // There 4 possible altera_syncram instantiations. The different settings are                                                             //
  // summarized below.  Every other port is the same among all instantiations.                                                              //
  // In BIDIR_DUAL_PORT mode or MLAB block type, addressstall_a and                                                                         //
  // addressstall_b should not be connected because they are not allowed on                                                                 //
  // Stratix 10 for BIDIR_DUAL_PORT and on Arria 10 MLABs.                                                                                  //
  //                                                                                                                                        //
  // -------------------------------------------------------------------------------------------------------------------------------------- //
  // ECC   Mode                        use_byteena   | byteena_a      byteena_b      addressstall_a  addressstall_b                         //
  // -------------------------------------------------------------------------------------------------------------------------------------- //
  // ON     (BIDIR_DUAL_PORT or MLAB)  1              Connected       Connected      Not Connected   Not Connected                          //
  // ON    !(BIDIR_DUAL_PORT or MLAB)  1              connected       Connected      Connected       Connected                              //
  // ON     (BIDIR_DUAL_PORT or MLAB)  0              Not Connected   Not Connected  Not Connected   Not Connected                          //
  // ON    !(BIDIR_DUAL_PORT or MLAB)  0              Not Connected   Not Connected  Connected       Connected                              //
  // -------------------------------------------------------------------------------------------------------------------------------------- //
  //                                                                                                                                        //
  // If ECC is OFF                                                                                                                          //
  // There are 8 possible altera_syncram instantiations. The different                                                                      //
  // settings are summarized below. Every other port is the same among all                                                                  //
  // instantiations.  the default value for width_byteena_a and width_byteena_b                                                             //
  // is 1. In BIDIR_DUAL_PORT mode or MLAB block type, addressstall_a and                                                                   //
  // addressstall_b should not be connected because they are not allowed on                                                                 //
  // Stratix 10 for BIDIR_DUAL_PORT and on Arria 10 MLABs.                                                                                  //
  //                                                                                                                                        //
  // -------------------------------------------------------------------------------------------------------------------------------------- //
  // ECC   Mode                        width_byteena_a   width_byteena_b   | byteena_a      byteena_b      addressstall_a  addressstall_b   //
  // -------------------------------------------------------------------------------------------------------------------------------------- //
  // OFF    (BIDIR_DUAL_PORT or MLAB)  ==1               ==1                 Not Connected  Not Connected  Not Connected   Not Connected    //
  // OFF    (BIDIR_DUAL_PORT or MLAB)  !=1               !=1                 Connected      Connected      Not Connected   Not Connected    //
  // OFF    (BIDIR_DUAL_PORT or MLAB)  !=1               ==1                 Connected      Not Connected  Not Connected   Not Connected    //
  // OFF    (BIDIR_DUAL_PORT or MLAB)  ==1               !=1                 Not Connected  Connected      Not Connected   Not Connected    //
  // OFF   !(BIDIR_DUAL_PORT or MLAB)  ==1               ==1                 Not connected  Not Connected  Connected       Connected        //
  // OFF   !(BIDIR_DUAL_PORT or MLAB)  !=1               !=1                 Connected      Connected      Connected       Connected        //
  // OFF   !(BIDIR_DUAL_PORT or MLAB)  !=1               ==1                 Connected      Not Connected  Connected       Connected        //
  // OFF   !(BIDIR_DUAL_PORT or MLAB)  ==1               !=1                 Not Connected  Connected      Connected       Connected        //
  // -------------------------------------------------------------------------------------------------------------------------------------- //
  //                                                                                                                                        //
  // Add additional parameter do_not_connect_read_enable which will connect the rden ports to                                               //
  // 1'b1 if enabled to ensure AUTO local memory can go into MLABs for A10 and                                                              //
  // older. The above cases are repeated for do_not_connect_read_enable = 0 and do_not_connect_read_enable = 1                              //
  //                                                                                                                                        //
  // -------------------------------------------------------------------------------------------------------------------------------------- //
  // do_not_connect_read_enable      rden_a      rden_b                                                                                     //
  // -------------------------------------------------------------------------------------------------------------------------------------- //
  // 0                               rden_a      rden_b                                                                                     //
  // 1                               1'b1        1'b1                                                                                       //
  // -------------------------------------------------------------------------------------------------------------------------------------- //
  //                                                                                                                                        //
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // All the common parameters for all the cases
  `define COMMON_ALTERA_SYNCRAM_PARAMS \
         .width_a                                (I_MEMORY_WIDTH_A),\
         .widthad_a                              (widthad_a),\
         .widthad2_a                             (widthad2_a),\
         .numwords_a                             (numwords_a),\
         .outdata_reg_a                          (outdata_reg_a),\
         .address_aclr_a                         (address_aclr_a),\
         .outdata_aclr_a                         (outdata_aclr_a),\
         .width_byteena_a                        (I_WIDTH_BYTEENA_A),\
         .width_b                                (I_MEMORY_WIDTH_B),\
         .widthad_b                              (widthad_b),\
         .widthad2_b                             (widthad2_b),\
         .numwords_b                             (numwords_b),\
         .rdcontrol_reg_b                        (rdcontrol_reg_b),\
         .address_reg_b                          (address_reg_b),\
         .outdata_reg_b                          (outdata_reg_b),\
         .outdata_aclr_b                         (outdata_aclr_b),\
         .indata_reg_b                           (indata_reg_b),\
         .byteena_reg_b                          (byteena_reg_b),\
         .address_aclr_b                         (address_aclr_b),\
         .width_byteena_b                        (I_WIDTH_BYTEENA_B),\
         .clock_enable_input_a                   (clock_enable_input_a),\
         .clock_enable_output_a                  (clock_enable_output_a),\
         .clock_enable_input_b                   (clock_enable_input_b),\
         .clock_enable_output_b                  (clock_enable_output_b),\
         .clock_enable_core_a                    (clock_enable_core_a),\
         .clock_enable_core_b                    (clock_enable_core_b),\
         .read_during_write_mode_port_a          (read_during_write_mode_port_a),\
         .read_during_write_mode_port_b          (read_during_write_mode_port_b),\
         .read_during_write_mode_mixed_ports     (read_during_write_mode_mixed_ports),\
         .enable_ecc                             ("FALSE"),\
         .width_eccstatus                        (2),\
         .ecc_pipeline_stage_enabled             ("FALSE"),\
         .outdata_sclr_a                         (outdata_sclr_a),\
         .outdata_sclr_b                         (outdata_sclr_b),\
         .enable_ecc_encoder_bypass              ("FALSE"),\
         .enable_coherent_read                   (enable_coherent_read),\
         .enable_force_to_zero                   (enable_force_to_zero),\
         .operation_mode                         (operation_mode),\
         .byte_size                              (byte_size),\
         .ram_block_type                         (ram_block_type),\
         .init_file                              (init_file),\
         .init_file_layout                       (init_file_layout),\
         .maximum_depth                          (maximum_depth),\
         .intended_device_family                 (intended_device_family),\
         .lpm_hint                               (lpm_hint),\
         .lpm_type                               (lpm_type),\
         .implement_in_les                       (implement_in_les),\
         .power_up_uninitialized                 (power_up_uninitialized)

  // All the common ports for all 8 cases
  `define COMMON_ALTERA_SYNCRAM_PORTS \
         .wren_a               (wren_a),\
         .wren_b               (wren_b),\
         .data_a               (i_codeword_wr_a),\
         .data_b               (i_codeword_wr_b),\
         .address_a            (address_a),\
         .address_b            (address_b),\
         .clock0               (clock0),\
         .clock1               (i_clock1),\
         .clocken0             (clocken0),\
         .clocken1             (i_clocken1),\
         .aclr0                (aclr0_int),\
         .aclr1                (aclr1_int),\
         .q_a                  (i_codeword_rd_a),\
         .q_b                  (i_codeword_rd_b),\
         .sclr                 (sclr_int),\
         .eccstatus            (),\
         .eccencbypass         (),\
         .eccencparity         (),\
         .address2_a           (),\
         .address2_b           (),\
         .clocken2             (),\
         .clocken3             ()

  generate
    // define read enable ports separately as rden_b does not work with MLABs
    // if this wrapper is being called from acl_mem1x then do not instaniate
    // read enables as they are never used (tied to 1)
    if (do_not_connect_read_enable == 1) begin
      // ECC ON, do_not_connect_read_enable = 1
      if (enable_ecc == "TRUE") begin
        if (use_byteena == 1) begin
          if (operation_mode == "BIDIR_DUAL_PORT" || ram_block_type == "MLAB" || do_not_connect_addressstall == 1) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            (i_byteena_a),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
          else begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            (i_byteena_a),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
        end
        else begin
          if (operation_mode == "BIDIR_DUAL_PORT" || ram_block_type == "MLAB" || do_not_connect_addressstall == 1) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
              .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
          else begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
              .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
        end
      end

      // ECC OFF, do_not_connect_read_enable = 1
      else begin
        // First four cases: BIDIR_DUAL_PORT
        if (operation_mode == "BIDIR_DUAL_PORT" || ram_block_type == "MLAB" || do_not_connect_addressstall == 1) begin
          if ((width_byteena_a == 1) && (width_byteena_b == 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
             .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
             .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
             .addressstall_a       (),
             .addressstall_b       ()
            );
          end
          else if ((width_byteena_a != 1) && (width_byteena_b != 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            (i_byteena_a),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
          else if ((width_byteena_a != 1) && (width_byteena_b == 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            (i_byteena_a),
              .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
          else begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
         end

         // Second four cases: NOT BIDIR_DUAL_PORT
         else begin
          if ((width_byteena_a == 1) && (width_byteena_b == 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
             .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
             .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
             .addressstall_a       (addressstall_a),
             .addressstall_b       (addressstall_b)
            );
          end
          else if ((width_byteena_a != 1) && (width_byteena_b != 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            (i_byteena_a),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
          else if ((width_byteena_a != 1) && (width_byteena_b == 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            (i_byteena_a),
              .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
          else begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (1'b1),
              .rden_b               (1'b1),
              .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
        end
      end
    end
    // do_not_connect_read_enable = 0
    else begin
      if (enable_ecc == "TRUE") begin
        if (use_byteena == 1) begin
          if (operation_mode == "BIDIR_DUAL_PORT" || ram_block_type == "MLAB" || do_not_connect_addressstall == 1) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            (i_byteena_a),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
          else begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            (i_byteena_a),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
        end
        else begin
          if (operation_mode == "BIDIR_DUAL_PORT" || ram_block_type == "MLAB" || do_not_connect_addressstall == 1) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
              .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
          else begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
              .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
        end
      end

      // ECC OFF, do_not_connect_read_enable = 0
      else begin
        // First four cases: BIDIR_DUAL_PORT
        if (operation_mode == "BIDIR_DUAL_PORT" || ram_block_type == "MLAB" || do_not_connect_addressstall == 1) begin
          if ((width_byteena_a == 1) && (width_byteena_b == 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
             `COMMON_ALTERA_SYNCRAM_PORTS,
             .rden_a               (rden_a),
             .rden_b               (rden_b),
             .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
             .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
             .addressstall_a       (),
             .addressstall_b       ()
            );
          end
          else if ((width_byteena_a != 1) && (width_byteena_b != 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            (i_byteena_a),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
          else if ((width_byteena_a != 1) && (width_byteena_b == 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            (i_byteena_a),
              .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
          else begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (),
              .addressstall_b       ()
            );
          end
         end

         // Second four cases: NOT BIDIR_DUAL_PORT
         else begin
          if ((width_byteena_a == 1) && (width_byteena_b == 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
             `COMMON_ALTERA_SYNCRAM_PORTS,
             .rden_a               (rden_a),
             .rden_b               (rden_b),
             .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
             .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
             .addressstall_a       (addressstall_a),
             .addressstall_b       (addressstall_b)
            );
          end
          else if ((width_byteena_a != 1) && (width_byteena_b != 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            (i_byteena_a),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
          else if ((width_byteena_a != 1) && (width_byteena_b == 1)) begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            (i_byteena_a),
              .byteena_b            ({I_WIDTH_BYTEENA_B{1'b1}}),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
          else begin
            altera_syncram #( `COMMON_ALTERA_SYNCRAM_PARAMS ) altera_syncram_inst (
              `COMMON_ALTERA_SYNCRAM_PORTS,
              .rden_a               (rden_a),
              .rden_b               (rden_b),
              .byteena_a            ({I_WIDTH_BYTEENA_A{1'b1}}),
              .byteena_b            (i_byteena_b),
              .addressstall_a       (addressstall_a),
              .addressstall_b       (addressstall_b)
            );
          end
        end
      end
    end
  endgenerate

endmodule // acl_altera_syncram_wrapped

`default_nettype wire
