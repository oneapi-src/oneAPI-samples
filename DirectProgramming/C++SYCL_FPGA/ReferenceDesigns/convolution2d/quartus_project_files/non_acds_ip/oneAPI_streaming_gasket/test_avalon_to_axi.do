vlib work
vlog -reportprogress 300 -work work oneapi_avalon_to_axi_gasket.sv
vlog -reportprogress 300 -work work test_avalon_to_axi.sv
vsim work.oneapi_avalon_to_axi_gasket work.test_avalon_to_axi 
run -all
exit