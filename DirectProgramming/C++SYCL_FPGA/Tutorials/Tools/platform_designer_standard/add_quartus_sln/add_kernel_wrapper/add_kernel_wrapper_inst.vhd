	component add_kernel_wrapper is
		port (
			clk_clk            : in  std_logic                     := 'X'; -- clk
			exception_add_data : out std_logic_vector(63 downto 0);        -- data
			irq_add_irq        : out std_logic;                            -- irq
			reset_reset_n      : in  std_logic                     := 'X'  -- reset_n
		);
	end component add_kernel_wrapper;

	u0 : component add_kernel_wrapper
		port map (
			clk_clk            => CONNECTED_TO_clk_clk,            --           clk.clk
			exception_add_data => CONNECTED_TO_exception_add_data, -- exception_add.data
			irq_add_irq        => CONNECTED_TO_irq_add_irq,        --       irq_add.irq
			reset_reset_n      => CONNECTED_TO_reset_reset_n       --         reset.reset_n
		);

