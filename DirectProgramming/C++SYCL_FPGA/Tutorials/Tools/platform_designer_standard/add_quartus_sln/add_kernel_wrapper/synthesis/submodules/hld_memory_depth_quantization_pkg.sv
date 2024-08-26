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


// hld_memory_depth_quantization_pkg.sv
//
// Parameter computation helper functions for MLAB/M20K based IP
//
// The basic idea is that if you want to use an MLAB, you are going to use an
// entire MLAB regardless of whether you want 1 address or 32 addresses. In the
// case of a FIFO, there is a small area penalty for widening the address bus,
// however the extra pipeline elasticity that using all 32 addresses provides is
// most likely worth the few extra registers. Likewise for an M20K, no advantage
// to using less than 512 addresses. However, M20K can have different geometries
// so it might be better to use more. For example, for a 1-bit wide fifo, may as
// well use the 2048x10 configuration instead of 512x40. All geometries in this
// file are supported across all FPGA families that use M20K. As future work,
// one could add specialized support for geometries in A10 or older devices.

package hld_memory_depth_quantization_pkg;

    // Use the full depth available to MLAB or M20K.
    function automatic int quantizeRamDepth;
    input int depth;
    begin
        quantizeRamDepth =
            (depth <= 32)  ?                    32 : //fits into min depth MLAB
            (depth <= 512) ?                   512 : //fits into min depth M20K
                             ((depth+511)/512)*512 ; //round up to nearest multiple of 512
    end
    endfunction

    // If the M20K is narrow, can use a deeper depth.
    function automatic int quantizeRamDepthUsingWidth;
    input int depth, width;
    begin
        quantizeRamDepthUsingWidth =
            (depth <= 32)                  ?                    32 :    //fits into min depth MLAB
            (depth <= 2048 && width <= 10) ?                  2048 :    //fits into single M20K
            (depth <= 1024 && width <= 20) ?                  1024 :    //fits into single M20K
            (depth <= 512)                 ?                   512 :    //fits into min depth M20K
                                             ((depth+511)/512)*512 ;    //round up to nearest multiple of 512
    end
    endfunction

    // hld_fifo uses LFSRs for address, so it needs the full 2^N address backing.
    // This is also the case for acl_dcfifo.
    // First snap up to 32 or 512, then round up the nearest power of 2.
    function automatic int quantizeFifoDepth;
    input int depth;
    begin
        quantizeFifoDepth = 2 ** $clog2(quantizeRamDepth(depth));
    end
    endfunction

    // Same idea as above, but use a width-aware depth quantization.
    function automatic int quantizeFifoDepthUsingWidth;
    input int depth, width;
    begin
        quantizeFifoDepthUsingWidth = 2 ** $clog2(quantizeRamDepthUsingWidth(depth, width));
    end
    endfunction

endpackage
