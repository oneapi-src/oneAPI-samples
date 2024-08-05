add wave -group "Nios V" -radix hexadecimal -position insertpoint sim:/pd_system_tb/pd_system_inst/nios/*
add wave -group "Code & Data RAM" -radix hexadecimal -position insertpoint sim:/pd_system_tb/pd_system_inst/code_data_ram/*
add wave -group "Accelerator" -radix hexadecimal -position insertpoint sim:/pd_system_tb/pd_system_inst/simple_dma_accelerator/*

config wave -signalnamewidth 1 
