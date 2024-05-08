vlib work
vlog -reportprogress 300 -work work oneapi_avs_to_axs_pixel_gasket.sv
vlog -reportprogress 300 -work work test_avalon_to_axi.sv
vsim work.oneapi_avs_to_axs_pixel_gasket work.test_avalon_to_axi 
run -all
exit