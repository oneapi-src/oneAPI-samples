//  Copyright (c) 2023 Intel Corporation                                  
//  SPDX-License-Identifier: MIT                                          

module add (
    input wire i_clk,
    input wire reset_button_n,
    output logic fpga_led
  );  
  
  // invert reset_button_n and pipeline it a bit
  logic reset_button_d1;
  logic reset_button_d2;
  logic reset_button_d3;
  
  always @ (posedge i_clk)
  begin
    reset_button_d1 <= ~reset_button_n;
    reset_button_d2 <= reset_button_d1;
    reset_button_d3 <= reset_button_d2;
  end
  
  
  // register the signal used by the LED
  wire sort_done;
  always @(posedge i_clk)
  begin
	// led is inverted
    fpga_led <= ~sort_done;
  end
  
  add_kernel_wrapper u0 (
    .exception_add_data (),               //  output,  width = 64, exception_add_1.data
    .irq_add_irq        (sort_done),      //  output,   width = 1,       irq_add_1.irq
    .clk_clk            (i_clk),          //   input,   width = 1,             clk.clk
    .reset_reset        (reset_button_d3) //   input,   width = 1,           reset.reset
  );

endmodule
