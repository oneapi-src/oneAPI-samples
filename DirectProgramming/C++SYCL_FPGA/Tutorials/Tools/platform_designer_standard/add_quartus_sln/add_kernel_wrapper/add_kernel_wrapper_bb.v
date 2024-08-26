
module add_kernel_wrapper (
	clk_clk,
	exception_add_data,
	irq_add_irq,
	reset_reset_n);	

	input		clk_clk;
	output	[63:0]	exception_add_data;
	output		irq_add_irq;
	input		reset_reset_n;
endmodule
