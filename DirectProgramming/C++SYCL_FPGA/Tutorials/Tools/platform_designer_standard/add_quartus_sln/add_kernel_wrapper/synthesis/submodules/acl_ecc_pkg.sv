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


// acl_ecc_pkg.sv
//
// Parameter computation helper functions for ECC
//
// Intended for usage in:
// - acl_ecc_decoder.sv
// - acl_ecc_encoder.sv
// - any file that instantiates either of the above, including testbench for ecc

package acl_ecc_pkg;

    // How to compute the number of parity bits:
    // For example, suppose we had 20 bits of data, let's assume we can add parity bits and the total number of bits would still be within the same power-of-2 group, in this case 32
    // Under this assumption, dataWidth bits of data will require $clog2(dataWidth) Hamming parity bits and 1 overall parity bit, so the total number of bits is $clog2(dataWidth)+1+dataWidth
    // Now we check if the assumption was met, once we add the parity bits do we still stay within the same power-of-2 group: check if $clog2($clog2(dataWidth)+1+dataWidth) == $clog2(dataWidth)
    // If the assumption is met then our guess of the number of parity bits is correct, otherwise we are going up one power-of-2 so we need an extra parity bit
    // This logic breaks down at small width where we have to go up by 2 or more power-of-2 groups to fit the parity bits, so the cases for dataWidth up to 4 have been handled separately
    function automatic int getParityBits;
    input int dataWidth;
    begin
        getParityBits = (dataWidth==1) ? 3 : (dataWidth<=4) ? 4 : ( $clog2($clog2(dataWidth)+1+dataWidth) == $clog2(dataWidth)) ? ($clog2(dataWidth)+1) : ($clog2(dataWidth)+2);
    end
    endfunction

    // given the total data width and the desired group size, how many groups do we end up with?
    function automatic int getNumGroups;
    input int dataWidth, eccGroupSize;
    begin
        getNumGroups = (dataWidth + eccGroupSize - 1) / eccGroupSize;   //ceiling( dataWidth / eccGroupSize)
    end
    endfunction

    // given the total data width and the desired group size, what is the size of the last group?
    // all groups except possibly the last group will be of size eccGroupSize, the remainder goes in the last group
    // last group size can be as low as 1 and as high as eccGroupSize
    function automatic int getLastGroupSize;
    input int dataWidth, eccGroupSize;
    begin
        getLastGroupSize = dataWidth - ((getNumGroups(dataWidth,eccGroupSize)-1) * eccGroupSize);
    end
    endfunction

    // without ecc grouping (intended for secded_decoder and secded_encoder), determine the encoded width given the raw data width
    function automatic int getEncodedBits;
    input int dataWidth;
    begin
        getEncodedBits = getParityBits(dataWidth) + dataWidth;
    end
    endfunction

    // with ecc grouping (intended for acl_ecc_decoder and acl_ecc_encoder), determine the encoded width given the raw data width and ecc group size
    function automatic int getEncodedBitsEccGroup;
    input int dataWidth, eccGroupSize;
    begin
        //total bits           = raw bits  +  parity bits of full group   * number of full groups                     + parity bits of last group
        getEncodedBitsEccGroup = dataWidth + (getParityBits(eccGroupSize) * (getNumGroups(dataWidth,eccGroupSize)-1)) + getParityBits(getLastGroupSize(dataWidth,eccGroupSize));
    end
    endfunction

endpackage
