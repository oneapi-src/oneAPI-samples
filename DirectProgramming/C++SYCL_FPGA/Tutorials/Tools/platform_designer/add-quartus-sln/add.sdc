set_time_format -unit ns -decimal_places 3
create_clock -name i_clk -period 10 [get_ports {i_clk}]

set_false_path -from [get_ports {reset_button_n}] -to * 
set_false_path -from [get_ports {fpga_led}] -to *
set_false_path -from * -to [get_ports {fpga_led}]