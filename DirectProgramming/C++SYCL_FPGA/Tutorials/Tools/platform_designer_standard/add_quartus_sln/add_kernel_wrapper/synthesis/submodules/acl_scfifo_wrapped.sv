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
// ACL SCFIFO WRAPPED                                                                //
//                                                                                   //
// DESCRIPTION                                                                       //
// ===========                                                                       //
// This is a wrapper around the scfifo component and behaves in                      //
// the exact same way when the parameter enable_ecc is set to "FALSE". The           //
// module has the added (soft) ECC feature.                                          //
//                                                                                   //
// The module has the exact same interface except for                                //
//   * One addtional parameter:                                                      //
//     - "enable_ecc": accepts "TRUE" or "FALSE". The default is "FALSE"             //
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
// with any family in almost the exact same way as the scfifo IP.                    //
//                                                                                   //
///////////////////////////////////////////////////////////////////////////////////////

`default_nettype none

module acl_scfifo_wrapped
import acl_ecc_pkg::*;
#(
  // GLOBAL PARAMETER DECLARATION
  parameter lpm_width               = 1,             //Specifies the width of the data and q ports
  parameter lpm_widthu              = 1,             //Specifies the width of the usedw port
  parameter lpm_numwords            = 2,             //Specifies the depths of the FIFO you require. The value must be at least 4. The value assigned must comply to the following equation: 2^LPM_WIDTHU
  parameter lpm_showahead           = "OFF",         //Specifies whether the FIFO is in normal mode (OFF) or show-ahead mode (ON).
  parameter lpm_type                = "scfifo",      //Identifies the library of parameterized modules (LPM) entity name. The values are SCFIFO and DCFIFO.
  parameter lpm_hint                = "USE_EAB=ON",
  parameter intended_device_family  = "Arria 10",
  parameter underflow_checking      = "ON",
  parameter overflow_checking       = "ON",
  parameter allow_rwcycle_when_full = "OFF",
  parameter use_eab                 = "ON",          //Specifies whether or not the FIFO IP core is constructed using the RAM blocks. The values are ON or OFF.
  parameter add_ram_output_register = "OFF",         //Specifies whether to register the q output of the internal scfifo before going into the decoder.
  parameter almost_full_value       = 0,
  parameter almost_empty_value      = 0,
  parameter maximum_depth           = 0,
  parameter enable_ecc              = "FALSE"
)
(
  // INPUT PORT DECLARATION
  input wire    [lpm_width-1:0] data,           //Holds the data to be written in the FIFO IP core when the wrreq signal is asserted
  input wire                    clock,
  input wire                    wrreq,          //Assert this signal to request for a write operation
  input wire                    rdreq,          //Assert this signal to request for a read operation.
  input wire                    aclr,
  input wire                    sclr,

  // OUTPUT PORT DECLARATION
  output logic  [lpm_width-1:0] q,              //Shows the data read from the read request operation
  output logic [lpm_widthu-1:0] usedw,          //Asserted when the usedw signal is greater than or equal to the almost_full_value parameter.
  output logic                  full,           //When asserted, the FIFO IP core is considered full. Do not perform write request operation when the FIFO IP core is full.
  output logic                  empty,          //When asserted, the FIFO IP core is considered empty. Do not perform read request operation when the FIFO IP core is empty
  output logic                  almost_full,    //Asserted when the usedw signal is greater than or equal to the almost_full_value parameter.
  output logic                  almost_empty,   //Asserted when the usedw signal is less than the almost_empty_value parameter.
  output logic            [1:0] ecc_err_status  //10 = error detected and corrected (memory in not updated), 01 = error detected but uncorrectable
);

  // use localparams to calcultate internal parameters
  localparam int MAX_ECC_WIDTH = 32;  //this value should be ECC_GROUP_SIZE and can be swept to tradeoff fmax vs memory overhead. Do not set this value larger than 64 due to altecc implementation
  localparam int SCFIFO_WIDTH  = (enable_ecc == "TRUE") ? getEncodedBitsEccGroup(lpm_width, MAX_ECC_WIDTH) : lpm_width; // width of codewords

  // signals for making internal connections
  logic [SCFIFO_WIDTH-1:0]      codeword_wr;     // data to write into the memory, including any ECC bits
  logic [SCFIFO_WIDTH-1:0]      codeword_rd;     // data read out of the memory, including any ECC bits

  logic err_fatal;                               // ecc status signal for detected uncorrected error
  logic err_corrected;                           // ecc status signal for detected corrected error

  localparam int ERR_CORRECTED_PULSE_EXTENSION_WIDTH = 3;  // amount of pulse stretching for the "error corrected" signal.

  genvar i;
  generate
    if (enable_ecc == "TRUE") begin                           : GEN_ECC_ENABLED
      logic err_corrected_int;
      logic err_corrected_pulse_extended;

      logic err_fatal_int;
      logic err_fatal_latched;

      logic [ERR_CORRECTED_PULSE_EXTENSION_WIDTH-1:0] d;

       // instantiate the ECC encoder for port a
      acl_ecc_encoder #(
        .DATA_WIDTH                     (lpm_width),
        .ECC_GROUP_SIZE                 (MAX_ECC_WIDTH),
        .INPUT_PIPELINE_STAGES          (0),
        .OUTPUT_PIPELINE_STAGES         (0)
      ) altecc_encoder_inst_a (
         .clock                         (clock),
         .clock_enable                  (1'b1),
         .i_data                        (data),
         .o_encoded                     (codeword_wr)
      );

      // instantiate the ECC decoder for port a
      acl_ecc_decoder #(
         .DATA_WIDTH                    (lpm_width),
         .ECC_GROUP_SIZE                (MAX_ECC_WIDTH),
         .INPUT_PIPELINE_STAGES         (0),
         .OUTPUT_PIPELINE_STAGES        (0),
         .STATUS_PIPELINE_STAGES        (0)
      ) altecc_decoder_inst_a (
         .clock                         (clock),
         .clock_enable                  (1'b1),
         .i_encoded                     (codeword_rd),
         .o_single_error_corrected      (err_corrected_int),      //Flag signal to reflect the status of data received. Denotes single-bit error found and corrected. You can use the data because it has already been corrected.
         .o_double_error_detected       (err_fatal_int),          // Flag signal to reflect the status of data received. Denotes double-bit error found, but not corrected. You must not use the data if this signal is asserted.
         .o_data                        (q)
      );

      always_ff @(posedge clock or posedge aclr) begin

        if (aclr) begin
          err_fatal_latched            <= 1'b0;
          err_corrected_pulse_extended <= 1'b0;
          d                            <=  'b0;

        end else begin

          if (!empty) begin
             err_fatal_latched <= err_fatal_latched | err_fatal_int;

             d[0] <= err_corrected_int;
                for (int i = 1; i < ERR_CORRECTED_PULSE_EXTENSION_WIDTH; i++) begin : GEN_RANDOM_BLOCK_NAME_R64
                   d[i] <= d[i-1];
                end

             err_corrected_pulse_extended <= ( | d) | err_corrected_int;
          end

          if (sclr) begin
             err_fatal_latched            <= 1'b0;
             err_corrected_pulse_extended <= 1'b0;
             d                            <=  'b0;
          end
        end
      end

      assign err_fatal      = err_fatal_latched;
      assign err_corrected  = err_corrected_pulse_extended;
      assign ecc_err_status = {err_corrected,err_fatal};

    end else begin                                  : GEN_ECC_DISABLED   // soft_ecc is false

      assign err_corrected  = 1'b0;
      assign err_fatal      = 1'b0;
      assign ecc_err_status = {err_corrected,err_fatal};
      assign codeword_wr    = data;
      assign q              = codeword_rd;

    end

  endgenerate

  scfifo #(
    .add_ram_output_register       (add_ram_output_register),
    .lpm_numwords                  (lpm_numwords),
    .lpm_showahead                 (lpm_showahead),
    .lpm_type                      (lpm_type),
    .lpm_hint                      (lpm_hint),
    .lpm_width                     (SCFIFO_WIDTH),
    .lpm_widthu                    (lpm_widthu),
    .intended_device_family        (intended_device_family),
    .overflow_checking             (overflow_checking),
    .underflow_checking            (underflow_checking),
    .allow_rwcycle_when_full       (allow_rwcycle_when_full),
    .use_eab                       (use_eab),
    .almost_full_value             (almost_full_value),
    .almost_empty_value            (almost_empty_value),
    .maximum_depth                 (maximum_depth),
    .enable_ecc                    ("FALSE")
  ) scfifo_inst (
    .data                          (codeword_wr),
    .clock                         (clock),
    .wrreq                         (wrreq),
    .rdreq                         (rdreq),
    .aclr                          (aclr),
    .sclr                          (sclr),
    .q                             (codeword_rd),
    .usedw                         (usedw),
    .full                          (full),
    .empty                         (empty),
    .almost_full                   (almost_full),
    .almost_empty                  (almost_empty),
    .eccstatus                     ()
  );

endmodule // acl_scfifo_wrapped

`default_nettype wire
