vlib work
vlog -reportprogress 300 -work work oneapi_axs_to_avs_pixel_gasket.sv
vlog -reportprogress 300 -work work test_axi_to_avalon.sv
vsim work.oneapi_axs_to_avs_pixel_gasket work.test_axi_to_avalon 
run -all
exit