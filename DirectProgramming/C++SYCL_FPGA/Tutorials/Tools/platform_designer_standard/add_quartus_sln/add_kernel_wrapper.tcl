# qsys scripting (.tcl) file for add_kernel_wrapper
package require -exact qsys 16.0

create_system {add_kernel_wrapper}

set_project_property DEVICE_FAMILY {Cyclone V}
set_project_property DEVICE {5CSEMA5F31C6}
set_project_property HIDE_FROM_IP_CATALOG {false}

# Instances and instance parameters
# (disabled instances are intentionally culled)
add_instance add_report_di_0 add_report_di 1.0

add_instance clk_0 clock_source 23.1
set_instance_parameter_value clk_0 {clockFrequency} {50000000.0}
set_instance_parameter_value clk_0 {clockFrequencyKnown} {1}
set_instance_parameter_value clk_0 {resetSynchronousEdges} {NONE}

add_instance master_0 altera_jtag_avalon_master 23.1
set_instance_parameter_value master_0 {FAST_VER} {0}
set_instance_parameter_value master_0 {FIFO_DEPTHS} {2}
set_instance_parameter_value master_0 {PLI_PORT} {50000}
set_instance_parameter_value master_0 {USE_PLI} {0}

# exported interfaces
add_interface clk clock sink
set_interface_property clk EXPORT_OF clk_0.clk_in
add_interface exception_add conduit end
set_interface_property exception_add EXPORT_OF add_report_di_0.device_exception_bus
add_interface irq_add interrupt sender
set_interface_property irq_add EXPORT_OF add_report_di_0.kernel_irqs
add_interface reset reset sink
set_interface_property reset EXPORT_OF clk_0.clk_in_reset

# connections and connection parameters
add_connection clk_0.clk add_report_di_0.clock

add_connection clk_0.clk master_0.clk

add_connection clk_0.clk_reset add_report_di_0.resetn

add_connection clk_0.clk_reset master_0.clk_reset

add_connection master_0.master add_report_di_0.csr_ring_root_avs
set_connection_parameter_value master_0.master/add_report_di_0.csr_ring_root_avs arbitrationPriority {1}
set_connection_parameter_value master_0.master/add_report_di_0.csr_ring_root_avs baseAddress {0x0000}
set_connection_parameter_value master_0.master/add_report_di_0.csr_ring_root_avs defaultConnection {0}

# interconnect requirements
set_interconnect_requirement {$system} {qsys_mm.clockCrossingAdapter} {HANDSHAKE}
set_interconnect_requirement {$system} {qsys_mm.enableEccProtection} {FALSE}
set_interconnect_requirement {$system} {qsys_mm.insertDefaultSlave} {FALSE}
set_interconnect_requirement {$system} {qsys_mm.maxAdditionalLatency} {1}

save_system {add_kernel_wrapper.qsys}
