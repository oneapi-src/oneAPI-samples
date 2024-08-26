//// (c) 1992-2024 Intel Corporation.                            
// Intel, the Intel logo, Intel, MegaCore, NIOS II, Quartus and TalkBack words    
// and logos are trademarks of Intel Corporation or its subsidiaries in the U.S.  
// and/or other countries. Other marks and brands may be claimed as the property  
// of others. See Trademarks on intel.com for full list of Intel trademarks or    
// the Trademarks & Brands Names Database (if Intel) or See www.Intel.com/legal (if Altera) 
// Your use of Intel Corporation's design tools, logic functions and other        
// software and tools, and its AMPP partner logic functions, and any output       
// files any of the foregoing (including device programming or simulation         
// files), and any associated documentation or information are expressly subject  
// to the terms and conditions of the Altera Program License Subscription         
// Agreement, Intel MegaCore Function License Agreement, or other applicable      
// license agreement, including, without limitation, that your use is for the     
// sole purpose of programming logic devices manufactured by Intel and sold by    
// Intel or its authorized distributors.  Please refer to the applicable          
// agreement for further details.                                                 


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Objective:
//  Cause a compile error if a module is instantiated with illegal parameter values. This must work in both simulation and Quartus.
//
//  Notes:
//  As of SystemVerilog 2009, $fatal() can be used in synthesizable code as an elaboration task, however many tools lack support for this
// newer specification. By using "enable_verilog_initial_construct", $fatal() ends up being treated as a system task instead of an elaboration
// task. This is supported in Quartus standard and Quartus pro, as well as pretty much every simulator (tested on Modelsim and VCS).
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

`ifndef ACL_PARAMETER_ASSERT_SVH
`define ACL_PARAMETER_ASSERT_SVH

//beware Quartus standard does not support line breaks e.g. splitting the macro across several lines using backslash
`define ACL_PARAMETER_ASSERT(COND) (* enable_verilog_initial_construct *) initial begin if (!(COND)) $fatal(1, "illegal parameterization, expecting %s", `"COND`"); end

//if COND contains strings, the stringify of COND will get messed up, so provide your own message
`define ACL_PARAMETER_ASSERT_MESSAGE(COND, MESSAGE) (* enable_verilog_initial_construct *) initial begin if (!(COND)) $fatal(1, "illegal parameterization, %s", MESSAGE); end

`endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Usage:
//  `include "acl_parameter_assert.svh"
//  `ACL_PARAMETER_ASSERT(your_condition)
//  `ACL_PARAMETER_ASSERT_MESSAGE(your_condition, your_message)
//  
//  Notes:
//  - your_condition should check for a legal parameterization
//  - no semicolon at the end
//  - if strings are used in your_condition, must use ACL_PARAMETER_ASSERT_MESSAGE
//  
//  Examples:
//  `ACL_PARAMETER_ASSERT(MY_PARAMETER > 100)
//  `ACL_PARAMETER_ASSERT(SMALL_PARAM < LARGE_PARAM)
//  `ACL_PARAMETER_ASSERT_MESSAGE(STRING_PARAM == "some_value" || STRING_PARAM == "other_value", "STRING_PARAM must be some_value or other_value")
//
//  Typical error message (assuming it happens on line 123 in your_file.sv):
//  - Quartus standard (20.1 release):
//    - Error (10917): SystemVerilog $fatal at your_file.sv(123): 1illegal parameterization, expecting %sMY_PARAMETER > 100
//    - Note there is a known issue that $fatal() does not parse properly, however the message is sufficiently human readable
//  - Quartus pro (20.1 release):
//    - Error(21491): Verilog HDL error at your_file.sv(123): $fatal : illegal parameterization, expecting MY_PARAMETER > 100
//  - VCS (2019.06-SP1):
//    - illegal parameterization, expecting MY_PARAMETER > 100
//    - $finish called from file "/your/path/goes/here/your_file.sv", line 123.
//  - Modelsim SE (2020.1):
//    - Fatal: illegal parameterization, expecting MY_PARAMETER > 100
//    - Time: 0 ps  Scope: tb.dut File: /your/path/goes/here/your_file.sv Line: 123
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
