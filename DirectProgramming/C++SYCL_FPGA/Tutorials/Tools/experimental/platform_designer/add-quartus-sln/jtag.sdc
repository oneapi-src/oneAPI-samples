#****************************************************************************
#
# SPDX-License-Identifier: MIT-0
# Copyright(c) 2015-2023 Intel Corporation.
#
#****************************************************************************
#
# Sample SDC for A10 GHRD. Targeting JTAG.
#
#****************************************************************************
# For USB BlasterII running at 24MHz or 41.666 ns period
set t_period 41.666
create_clock -name {altera_reserved_tck} -period $t_period  [get_ports {altera_reserved_tck}]
set_clock_groups -asynchronous -group {altera_reserved_tck}

#Datasheet parameters from UBII IP on EPM570F100C5
#TCO/TSU/TH are measured w.r.t usb_clk inside UBII IP which is used to generate TCK signal
set tck_blaster_tco_max 14.603
set tck_blaster_tco_min 14.603
set tdi_blaster_tco_max 8.551
set tdi_blaster_tco_min 8.551
set tms_blaster_tco_max 9.468
set tms_blaster_tco_min 9.468

#In bitbang mode, TDO is sampled through MAX at FX2
set tdo_blaster_tpd_max 10.718
set tdo_blaster_tpd_min 10.718
set fx2_pb0_trace_max 0.152
set fx2_pb0_trace_min 0.152

#Cable delays are from USB Blaster II
#TCK
set tck_cable_max 11.627
set tck_cable_min 10.00
#*USER MODIFY* This depends on the trace length from JTAG 10-pin header to FPGA on board
set tck_header_trace_max 0.5
set tck_header_trace_min 0.1

#TMS
set tms_cable_max 11.627
set tms_cable_min 10.0
#*USER MODIFY* This depends on the trace length from JTAG 10-pin header to FPGA on board
set tms_header_trace_max 0.5
set tms_header_trace_min 0.1

#TDI
set tdi_cable_max 11.627
set tdi_cable_min 10.0
#*USER MODIFY* This depends on the trace length from JTAG 10-pin header to FPGA on board
set tdi_header_trace_max 0.5
set tdi_header_trace_min 0.1

#TDO
set tdo_cable_max 11.627
set tdo_cable_min 10.0
#*USER MODIFY* This depends on the trace length from JTAG 10-pin header to FPGA on board
set tdo_header_trace_max 0.5
set tdo_header_trace_min 0.1

derive_clock_uncertainty

#TMS
set tms_in_max [expr {$tms_cable_max + $tms_header_trace_max + $tms_blaster_tco_max - $tck_blaster_tco_min - $tck_cable_min - $tck_header_trace_min }]
set tms_in_min [expr {$tms_cable_min + $tms_header_trace_min + $tms_blaster_tco_min - $tck_blaster_tco_max - $tck_cable_max - $tck_header_trace_max }]
set_input_delay -add_delay -clock_fall -clock altera_reserved_tck -max $tms_in_max [get_ports {altera_reserved_tms}]
set_input_delay -add_delay -clock_fall -clock altera_reserved_tck -min $tms_in_min [get_ports {altera_reserved_tms}]

#TDI
set tdi_in_max [expr {$tdi_cable_max + $tdi_header_trace_max + $tdi_blaster_tco_max - $tck_blaster_tco_min - $tck_cable_min - $tck_header_trace_min }]
set tdi_in_min [expr {$tdi_cable_min + $tdi_header_trace_min + $tdi_blaster_tco_min - $tck_blaster_tco_max - $tck_cable_max - $tck_header_trace_max }]
set_input_delay -add_delay -clock_fall -clock altera_reserved_tck -max $tdi_in_max [get_ports {altera_reserved_tdi}]
set_input_delay -add_delay -clock_fall -clock altera_reserved_tck -min $tdi_in_min [get_ports {altera_reserved_tdi}]

#TDO Timing in Bitbang Mode
#TDO timing delays must take into account the TCK delay from the Blaster to the FPGA TCK input pin
set tdo_out_max [expr {$tdo_cable_max + $tdo_header_trace_max + $tdo_blaster_tpd_max + $fx2_pb0_trace_max + $tck_blaster_tco_max + $tck_cable_max + $tck_header_trace_max }]
set tdo_out_min [expr {$tdo_cable_min + $tdo_header_trace_min + $tdo_blaster_tpd_min + $fx2_pb0_trace_min + $tck_blaster_tco_min + $tck_cable_min + $tck_header_trace_min }]

#TDO does not latch inside the USB Blaster II at the rising edge of TCK, it actually is passed through to the Cypress FX2 and is latched 3 FX2 cycles later (equivalent to 1.5 JTAG cycles)
set_output_delay -add_delay -clock altera_reserved_tck -max $tdo_out_max [get_ports {altera_reserved_tdo}]
set_output_delay -add_delay -clock altera_reserved_tck -min $tdo_out_min [get_ports {altera_reserved_tdo}]

set_multicycle_path -setup -end 2 -from * -to [get_ports {altera_reserved_tdo}]

