//  Copyright (c) 2023 Intel Corporation                                  
//  SPDX-License-Identifier: MIT                                          

module add (
    input wire i_clk,
    input wire reset_button_n,
    output logic fpga_led
  );

  // reset synchronizer
  logic [2:0] sync_resetn;
  always @(posedge i_clk or negedge reset_button_n) begin
    if (!reset_button_n) begin
      sync_resetn <= 3'b0;
    end else begin
      sync_resetn <= {sync_resetn[1:0], 1'b1};
    end
  end
  logic synchronized_resetn;
  assign synchronized_resetn = sync_resetn[2];
  
  // register the signal used by the LED
  wire sort_done;
  always @(posedge i_clk)
  begin
    fpga_led <= sort_done;
  end
  
  add_kernel_wrapper u0 (
    .exception_add_data (),                   //  output,  width = 64, exception_add_1.data
    .irq_add_irq        (sort_done),          //  output,   width = 1,       irq_add_1.irq
    .clk_clk            (i_clk),              //   input,   width = 1,             clk.clk
    .reset_reset        (synchronized_resetn) //   input,   width = 1,           reset.reset
  );

endmodule
