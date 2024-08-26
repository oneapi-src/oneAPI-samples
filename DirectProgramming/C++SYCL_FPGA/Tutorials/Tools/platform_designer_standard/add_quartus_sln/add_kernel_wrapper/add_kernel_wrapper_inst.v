	add_kernel_wrapper u0 (
		.clk_clk            (<connected-to-clk_clk>),            //           clk.clk
		.exception_add_data (<connected-to-exception_add_data>), // exception_add.data
		.irq_add_irq        (<connected-to-irq_add_irq>),        //       irq_add.irq
		.reset_reset_n      (<connected-to-reset_reset_n>)       //         reset.reset_n
	);

