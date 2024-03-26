vlib work
vlog -reportprogress 300 -work work oneapi_axi_to_avalon_gasket.sv
vlog -reportprogress 300 -work work test_axi_to_avalon.sv
vsim work.oneapi_axi_to_avalon_gasket work.test_axi_to_avalon 
run -all
exit