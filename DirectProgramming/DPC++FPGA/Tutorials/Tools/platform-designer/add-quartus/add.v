//  Copyright (c) 2022 Intel Corporation                                  
//  SPDX-License-Identifier: MIT                                          

module add (
    input wire i_clk,
    input wire reset_button_n,
    output reg fpga_led
  );

  // pipeline the `reset` signal so the fitter can move logic around on the
  // chip. Also synchronize `reset` signal to clock. Debouncing logic would be
  // good, but it is omitted for simplicity.
  reg reset_button_d1;
  reg reset_button_d2;

  always @ (posedge i_clk)
  begin
    reset_button_d1 <= ~reset_button_n;
    reset_button_d2 <= reset_button_d1;
  end
  
  // register the signal used by the LED
  wire sort_done;
  always @(posedge i_clk)
  begin
    fpga_led <= sort_done;
  end

  assign reset = reset_button_d2;
  
  wire irq_members;
  wire irq_hostpipes;
  
  assign sort_done = irq_members | irq_hostpipes;
  
		add_kernels u0 (
		.exception_add_hostpipes_1_data (),               //  output,  width = 64, exception_add_hostpipes_1.data
		.irq_add_hostpipes_1_irq        (irq_hostpipes),  //  output,   width = 1,       irq_add_hostpipes_1.irq
		.exception_add_members_1_data   (),               //  output,  width = 64,   exception_add_members_1.data
		.irq_add_members_1_irq          (irq_members),    //  output,   width = 1,         irq_add_members_1.irq
		.clk_clk                        (i_clk),          //   input,   width = 1,                       clk.clk
		.reset_reset                    (reset)           //   input,   width = 1,                     reset.reset
	);





endmodule
