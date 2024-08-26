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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                          //
//  ACL LATENCY ONE RAM FIFO                                                                                                                                                                                                //
//  Designed and optimized by: Jason Thong                                                                                                                                                                                  //
//                                                                                                                                                                                                                          //
//  DESCRIPTION                                                                                                                                                                                                             //
//  ===========                                                                                                                                                                                                             //
//  This fifo has a write to read latency of 1. Low latency fifos are often implemented with registers as storage, but this uses lots of logic as the fifo capacity increases. When the fifo latency is lower               //
//  than the read latency from embedded memory, one must add bypass logic so that newly written into the fifo can appear at the output before it is readable from the memory. That is what this fifo does.                  //
//                                                                                                                                                                                                                          //
//  REQUIRED FILES                                                                                                                                                                                                          //
//  ==============                                                                                                                                                                                                          //
//  - acl_latency_one_ram_fifo.sv                                                                                                                                                                                           //
//  - acl_lfsr.sv                                                                                                                                                                                                           //
//  - acl_tessellated_incr_decr_threshold.sv                                                                                                                                                                                //
//  - acl_reset_handler.sv                                                                                                                                                                                                  //
//                                                                                                                                                                                                                          //
//  RELATIONSHIP TO ACL_MID_SPEED_FIFO                                                                                                                                                                                      //
//  ==================================                                                                                                                                                                                      //
//  This fifo is largely derived from acl_mid_speed_fifo including all its caveats of having different implementations for M20K vs MLAB. Conceptually, acl_mid_speed_fifo can be thought of as having ram storage with      //
//  unregistered read data from the memory and 1 prefetch stage on the read side. In the implementation that prefetch stage is pushed inside the ram. To reduce the write to read latency from 3 to 2, all that is needed   //
//  is "new data mode" for mixed port reading during write. In other words, if the write address and the read address are the same, the read data should be the newly written data. This can be implemented with external   //
//  logic that forwards the newly written data to the output. To then reduce the write to read latency to 1, the fifo output must capture newly written data when the fifo is empty. The structure looks like this:         //
//                                                                                                                                                                                                                          //
//                                              ^                                                                                                                                                                           //
//                  +--------------+            |\                                                                                                                                                                          //
//             +----+ data_in_prev +------------+ \                                                                                                                                                                         //
//             |    +--------------+            |  \                                                                                                                                                                        //
//             |                                |   \                                                                                                                                                                       //
//             +--------------------------------+    |                                                                                                                                                                      //
//             |                                |    |      +----------+                                                                                                                                                    //
//  data_in ---+    +--------------------+      |    +------+ data_out +-------                                                                                                                                             //
//             |    |    M20K or MLAB    |      |    |      +----------+                                                                                                                                                    //
//             |    |                    |      |    |                                                                                                                                                                      //
//             +----+- write_data        |      |   /                                                                                                                                                                       //
//                  |                    |      |  /                                                                                                                                                                        //
//                  |   read_data_unreg -+------+ /                                                                                                                                                                         //
//                  +--------------------+      |/                                                                                                                                                                          //
//                                              v                                                                                                                                                                           //
//                                                                                                                                                                                                                          //
//  The output from the M20K or MLAB is unregistered, and a 3:1 mux is needed before the data_out register. To improve fmax, we need to register the output of the M20K or MLAB, which will result in unregistered          //
//  data_out. For M20K, registered read data means that register can be packed inside the M20K itself so it no longer consumes ALM registers, which results in an area savings.                                             //
//                                                                                                                                                                                                                          //
//                                      ^                                                                                                                                                                                   //
//                  +--------------+    |\                                                                                                                                                                                  //
//             +----+ data_in_prev +----+ \                         ^                                                                                                                                                       //
//             |    +--------------+    |  |    +-------------+     |\                                                                                                                                                      //
//             |                        |  +----+ data_in_mux +-----+ \                                                                                                                                                     //
//             +------------------------+ /     +-------------+     |  \                                                                                                                                                    //
//             |                        |/                          |   \                                                                                                                                                   //
//             |                        v                           |    |                                                                                                                                                  //
//             |                                                    |    |                                                                                                                                                  //
//  data_in ---+    +--------------------+                          |    +------- data_out (unregistered)                                                                                                                   //
//             |    |    M20K or MLAB    |                          |    |                                                                                                                                                  //
//             |    |                    |                          |    |                                                                                                                                                  //
//             +----+- write_data        |                          |   /                                                                                                                                                   //
//                  |                    |                          |  /                                                                                                                                                    //
//                  |     read_data_reg -+--------------------------+ /                                                                                                                                                     //
//                  +--------------------+                          |/                                                                                                                                                      //
//                                                                  v                                                                                                                                                       //
//                                                                                                                                                                                                                          //
//  REGISTERED_DATA_OUT_COUNT                                                                                                                                                                                               //
//  =========================                                                                                                                                                                                               //
//  We support the ability to have some data_out bit registers while others are unergistered. The fifo favors unregistered data_out, however if there are a few bits of data_out that feed some timing critical control     //
//  path, then one can register just those bits. If we need registered data_out (which requires unregistered data from the M20K or MLAB), the fmax penalty is less severe if we only need it for a few bits rather than     //
//  all bits of data_out. However there is a quantization problem: for a given physical M20K or MLAB, we either have to register all output data or unregister all of them. Our implementation will not increase RAM        //
//  usage due to this, what actually happens is REGISTERED_DATA_OUT_COUNT specifies the minimum number of bits in data_out that are registered.                                                                             //
//                                                                                                                                                                                                                          //
//  Here are some examples to illustrate what will happen. Supposed we are using MLAB with a physical width of 20.                                                                                                          //
//                                                                                                                                                                                                                          //
//  Example 1: DATA_WIDTH = 32, REGISTERED_DATA_OUT_COUNT = 4                                                                                                                                                               //
//  Based on DATA_WIDTH we need 2 physical MLABs. Based on REGISTERED_DATA_OUT_COUNT we need 1 physical MLAB that results in registered data_out (output of fifo, not MLAB). Therefore the remaining 1 MLAB can produce     //
//  unregistered data_out, and we want to maximize uasge of this. The final configuration will be 12 bits of regsitered data_out (>= 4 that the user asked for) and 20 bits of unregistered data_out. To get exactly 4      //
//  bits of registered data_out would have required 3 physical MLABs and we do not implement this.                                                                                                                          //
//                                                                                                                                                                                                                          //
//  Example 3: DATA_WIDTH = 64, REGISTERED_DATA_OUT_COUNT = 8                                                                                                                                                               //
//  Based on DATA_WIDTH we need 4 physical MLABs. Based on REGISTERED_DATA_OUT_COUNT we need 1 physical MLAB that results in registered data_out. Therefore the 3 remaining MLABs can produce unregistered data_out,        //
//  and this is enough MLABs to produce 56 bits of unregistered data_out. In this case the user will get exactly the requested 8 bits of registered data_out.                                                               //
//                                                                                                                                                                                                                          //
//  STALL_IN_EARLINESS AND VALID_IN_EARLINESS                                                                                                                                                                               //
//  =========================================                                                                                                                                                                               //
//  Similar to acl_low_latency_fifo, there are only two modes of earliness: none, or stall_in and valid_in both 1 clock early. For a latency one fifo, write_used_words == read_used_words, so in order to retime internal  //
//  occupancy tracking 1 clock earlier, we need both stall_in and valid_in to be early. This is intended for wide fifos in which we need more time to distribute control signals that fan-out to the entire data path.      //
//  Like acl_mid_speed_fifo, there is a very small area penalty for registering the most timing sensitive paths. For highest fmax, use STALL_IN_EARLINESS = 1, VALID_IN_EARLINESS = 1 and REGISTERED_DATA_OUT_COUNT = 0.    //
//                                                                                                                                                                                                                          //
//  RELATIONSHIP TO ACL_LOW_LATENCY_FIFO                                                                                                                                                                                    //
//  ====================================                                                                                                                                                                                    //
//  Functionally, this fifo can be used interchangeably with acl_low_latency_fifo. Both have a write to read latency of one clock cycle and have the same earliness optimizations. The main difference is the logic         //
//  utilization. This fifo uses memory as storage whereas acl_low_latency_fifo uses registers as storage. For shallow fifos, acl_low_latency_fifo is better. As the capacity increases, eventually it is cheaper to store   //
//  the data in memory instead. There is a simple heuristic in hld_fifo which estimates which fifo will result in lower area. Note that acl_low_latency_fifo has a much higher fmax, but typically most designs are not     //
//  fast enough to be limited by the lower fmax of acl_latency_one_ram_fifo.                                                                                                                                                //
//                                                                                                                                                                                                                          //
//  HLD_FIFO FEATURES                                                                                                                                                                                                       //
//  =================                                                                                                                                                                                                       //
//  This fifo is fully feature complete against all of the parameters exposed by hld_fifo. It follows the same reset scheme of asserting full and empty during reset.                                                       //
//                                                                                                                                                                                                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

`default_nettype none

module acl_latency_one_ram_fifo #(
    //basic fifo configuration
    parameter int WIDTH,                        // width of the data path through the fifo
    parameter int DEPTH,                        // capacity of the fifo, at least 1
    
    //occupancy
    parameter int ALMOST_EMPTY_CUTOFF = 0,      // almost_empty asserts if read_used_words <= ALMOST_EMPTY_CUTOFF, read_used_words increments when writes are visible on the read side, decrements when fifo is read
    parameter int ALMOST_FULL_CUTOFF = 0,       // almost_full asserts if write_used_words >= (DEPTH-ALMOST_FULL_CUTOFF), write_used_words increments when fifo is written to, decrements when fifo is read
    parameter int INITIAL_OCCUPANCY = 0,        // number of words in the fifo (write side occupancy) when it comes out of reset, note it still takes 5 clocks for this to become visible on the read side
    
    //reset configuration
    parameter bit ASYNC_RESET = 0,              // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET = 1,        // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
    parameter bit RESET_EVERYTHING = 0,         // intended for partial reconfig debug, set to 1 to reset every register (normally async reset excludes data path and sync reset additionally excludes some control signals)
    parameter bit RESET_EXTERNALLY_HELD = 1,    // set to 1 if resetn will be held for at least FOUR clock cycles, otherwise we will internally pulse stretch reset before consumption
    
    //ram implementation
    parameter string RAM_BLOCK_TYPE = "FIFO_TO_CHOOSE", // "MLAB" | "M20K" | "FIFO_TO_CHOOSE" -> different implementations for MLAB vs M20K, so cannot let quartus decide in case MLAB or M20K is not explicitly specified
    
    //special configurations for higher fmax / low area
    parameter int STALL_IN_EARLINESS = 0,       // how many clock cycles early is stall_in provided, fifo supports up to 2, setting this any higher results in registers to absorb the excess earliness
    parameter int VALID_IN_EARLINESS = 0,       // how many clock cycles early is valid_in provided, fifo can take advantage of 1 clock or valid_in earliness if stall_in is also at least 2 clocks early
    parameter int STALL_IN_OUTSIDE_REGS = 0,    // number of registers on the stall-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
    parameter int VALID_IN_OUTSIDE_REGS = 0,    // number of registers on the valid-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
    parameter int REGISTERED_DATA_OUT_COUNT = 0,// AT LEAST THIS MANY of the lower bits of data_out will be registered
    parameter bit NEVER_OVERFLOWS = 0,          // set to 1 to disable fifo's internal overflow protection, area savings by removing one incr/decr/thresh, stall_out still asserts during reset but won't mask valid_in
    
    //special features that typically have an fmax penalty
    parameter bit HOLD_DATA_OUT_WHEN_EMPTY = 0, // 0 means data_out can be x when fifo is empty, 1 means data_out will hold last value when fifo is empty (scfifo behavior, has fmax penalty)
    parameter bit WRITE_AND_READ_DURING_FULL = 0,//set to 1 to allow writing and reading while the fifo is full, this may have an fmax penalty, to compensate it is recommended to use this with NEVER_OVERFLOWS = 1
    
    //hidden parameters
    parameter int ZLRAM_RESET_RELEASE_DELAY_OVERRIDE = -1,  //DO NOT TOUCH, only acl_latency_zero_ram_fifo should set this
    
    //error correction code
    parameter enable_ecc = "FALSE"              // NOT IMPLEMENTED YET, see case:555783
)
(
    input  wire                 clock,
    input  wire                 resetn,         // see description above for different reset modes
    
    //write interface
    input  wire                 valid_in,       // upstream advertises it has data, a write happens when valid_in & ~stall_out -- this needs to be early if VALID_IN_EARLINESS >= 1
    input  wire     [WIDTH-1:0] data_in,        // data from upstream
    output logic                stall_out,      // inform upstream that we cannot accept data
    output logic                almost_full,    // asserts if write_used_words >= (DEPTH-ALMOST_FULL_CUTOFF)
    
    //read interface
    output logic                valid_out,      // advertise to downstream that we have data
    output logic    [WIDTH-1:0] data_out,       // data to downstream
    input  wire                 stall_in,       // downstream indicates it cannot accept data -- this needs to be early if STALL_IN_EARLINESS >= 1
    output logic                almost_empty,   // asserts if read_used_words <= ALMOST_EMPTY_CUTOFF
    output logic                forced_read_out,// indicates fifo is being read on current clock cycle, read data must be consumed or it will be lost, is a registered signal if STALL_IN_EARLINESS >= 1
    
    //signals intended only for acl_latency_zero_ram_fifo
    output logic                zlram_occ_gte_one_E,
    output logic                zlram_stall_out_E,
    
    //other
    output logic          [1:0] ecc_err_status  // NOT IMPLEMENTED YET, see case:555783
);
    
    //////////////////////////////////////
    //                                  //
    //  Sanity check on the parameters  //
    //                                  //
    //////////////////////////////////////
    
    // do not allow arbitrarily large amounts of earliness, as this delays the exit from reset "safe state"
    // the checks are done in Quartus pro and Modelsim, it is disabled in Quartus standard because it results in a syntax error (parser is based on an older systemverilog standard)
    // the workaround is to use synthesis translate to hide this from Quartus standard, ALTERA_RESERVED_QHD is only defined in Quartus pro, and Modelsim ignores the synthesis comment
    `ifdef ALTERA_RESERVED_QHD
    `else
    //synthesis translate_off
    `endif
    generate
    if (DEPTH < 1) begin
        $fatal(1, "acl_latency_one_ram_fifo: illegal value of DEPTH = %d, minimum allowed is 1\n", DEPTH);
    end
    if ((ALMOST_EMPTY_CUTOFF < 0) || (ALMOST_EMPTY_CUTOFF > DEPTH)) begin
        $fatal(1, "acl_latency_one_ram_fifo: illegal value of ALMOST_EMPTY_CUTOFF = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", ALMOST_EMPTY_CUTOFF, DEPTH);
    end
    if ((ALMOST_FULL_CUTOFF < 0) || (ALMOST_FULL_CUTOFF > DEPTH)) begin
        $fatal(1, "acl_latency_one_ram_fifo: illegal value of ALMOST_FULL_CUTOFF = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", ALMOST_FULL_CUTOFF, DEPTH);
    end
    if ((INITIAL_OCCUPANCY < 0) || (INITIAL_OCCUPANCY > DEPTH)) begin
        $fatal(1, "acl_latency_one_ram_fifo: illegal value of INITIAL_OCCUPANCY = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", INITIAL_OCCUPANCY, DEPTH);
    end
    if ((REGISTERED_DATA_OUT_COUNT < 0) || (REGISTERED_DATA_OUT_COUNT > WIDTH)) begin
        $fatal(1, "acl_latency_one_ram_fifo: illegal value of REGISTERED_DATA_OUT_COUNT = %d, minimum allowed is 0, maximum allowed is WIDTH = %d\n", REGISTERED_DATA_OUT_COUNT, WIDTH);
    end
    if ((STALL_IN_EARLINESS < 0) || (STALL_IN_EARLINESS > 10)) begin
        $fatal(1, "acl_latency_one_ram_fifo: illegal value of STALL_IN_EARLINESS = %d, minimum allowed is 0, maximum allowed is 10\n", STALL_IN_EARLINESS);
    end
    if ((VALID_IN_EARLINESS < 0) || (VALID_IN_EARLINESS > 10)) begin
        $fatal(1, "acl_latency_one_ram_fifo: illegal value of VALID_IN_EARLINESS = %d, minimum allowed is 0, maximum allowed is 10\n", VALID_IN_EARLINESS);
    end
    if (VALID_IN_OUTSIDE_REGS < 0 || VALID_IN_OUTSIDE_REGS > 1) begin
        $fatal(1, "acl_high_speed_fifo: illegal value of VALID_IN_OUTSIDE_REGS = %d, minimum allowed is 0, maximum allowed is 1\n", VALID_IN_OUTSIDE_REGS);
    end
    if (STALL_IN_OUTSIDE_REGS < 0 || STALL_IN_OUTSIDE_REGS > 1) begin
        $fatal(1, "acl_high_speed_fifo: illegal value of STALL_IN_OUTSIDE_REGS = %d, minimum allowed is 0, maximum allowed is 1\n", STALL_IN_OUTSIDE_REGS);
    end
    endgenerate
    `ifdef ALTERA_RESERVED_QHD
    `else
    //synthesis translate_on
    `endif
    
    
    
    //////////////////////////
    //                      //
    //  Parameter settings  //
    //                      //
    //////////////////////////
    
    // fifo configuration
    localparam int ADDR_RAW             = $clog2(DEPTH);
    localparam int ADDR                 = (ADDR_RAW < 2) ? 2 : ADDR_RAW;                                                            // minimum size of lfsr
    localparam bit USE_MLAB             = (RAM_BLOCK_TYPE == "MLAB") ? 1 : (RAM_BLOCK_TYPE == "M20K") ? 0 : (ADDR <= 5) ? 1 : 0;    //0 = mlab, 1 = m20k
    
    // ram configuration
    localparam int PHYSICAL_MLAB_WIDTH  = 20;
    localparam int PHYSICAL_M20K_WIDTH  = (DEPTH <= 512) ? 40 : (DEPTH <= 1024) ? 20 : 10;              //this is accurate for s10 but conservative for a10
    localparam int PHYSICAL_RAM_WIDTH   = (USE_MLAB) ? PHYSICAL_MLAB_WIDTH : PHYSICAL_M20K_WIDTH;
    localparam int LEFTOVER_UNREG       = (WIDTH - REGISTERED_DATA_OUT_COUNT) % PHYSICAL_RAM_WIDTH;     //once we pack all the unregistered data_out bits into full-width physical rams, how many bits are leftover?
    localparam int LEFTOVER_REG         = REGISTERED_DATA_OUT_COUNT % PHYSICAL_RAM_WIDTH;               //likewise for registered data_out bits
    localparam int ADD_TO_REG_DATA_OUT  = ((LEFTOVER_UNREG + LEFTOVER_REG) <= PHYSICAL_RAM_WIDTH) ? LEFTOVER_UNREG : 0; //if all the leftovers fit within one full-width physical ram, pack them together
    localparam int REG_DATA_OUT_COUNT   = REGISTERED_DATA_OUT_COUNT + ADD_TO_REG_DATA_OUT;              //total number of data_out bits that will be registered
    localparam int UNREG_DATA_OUT_COUNT = WIDTH - REG_DATA_OUT_COUNT;                                   //total number of data_out bits that will be unregistered
    
    // earliness configuration
    localparam int EARLY_MODE           = (STALL_IN_EARLINESS >= 1 && VALID_IN_EARLINESS >= 1) ? 1 : 0;
    localparam int EARLY_VALID          = EARLY_MODE;
    localparam int EXCESS_EARLY_VALID   = VALID_IN_EARLINESS - EARLY_VALID;
    localparam int EARLY_STALL          = EARLY_MODE;
    localparam int EXCESS_EARLY_STALL   = STALL_IN_EARLINESS - EARLY_STALL;
    
    // reset timing
    localparam int EXCESS_EARLY_STALL_WITH_EXT = EXCESS_EARLY_STALL + STALL_IN_OUTSIDE_REGS;    //early stall is affected by regs outside this module; account for effect on reset timing
    localparam int EXCESS_EARLY_VALID_WITH_EXT = EXCESS_EARLY_VALID + VALID_IN_OUTSIDE_REGS;    //early valid is affected by regs outisde this module; account for effect on reset timing
    localparam int FLUSH_EARLY_PIPES    = (EXCESS_EARLY_STALL_WITH_EXT > EXCESS_EARLY_VALID_WITH_EXT) ? EXCESS_EARLY_STALL_WITH_EXT : EXCESS_EARLY_VALID_WITH_EXT;  // clocks needs to flush excess earliness pipelines
    localparam int RESET_SYNC_DEPTH     = (SYNCHRONIZE_RESET) ? 3 : 0;                                                          // how many registers are added inside acl_reset_handler for synchronizing the reset
    localparam int RESET_PIPE_DEPTH     = 2;                                                                                    // how many pipeline stages we add to sclrn
    localparam int RESET_LATENCY        = (ASYNC_RESET || RESET_EVERYTHING) ? 0 : (RESET_SYNC_DEPTH + RESET_PIPE_DEPTH);        // how many clocks from the resetn input signal until the reset is consumed
    localparam int MIN_RESET_DELAY      = EARLY_MODE;                                                                           // internal occ tracking is 1 clock early when EARLY_MODE=1, 1 clock to propagate to stall_out
    localparam int RAW_RESET_DELAY      = FLUSH_EARLY_PIPES - RESET_LATENCY;                                                    // delay fifo exit from safe state if need more clocks to flush earliness than reset latency
    localparam int RESET_RELEASE_DELAY_PRE = (RAW_RESET_DELAY < MIN_RESET_DELAY) ? MIN_RESET_DELAY : RAW_RESET_DELAY;           // how many clocks late the fifo exits from safe state, excluding override from zlram
    localparam int RESET_RELEASE_DELAY  = (ZLRAM_RESET_RELEASE_DELAY_OVERRIDE != -1) ? ZLRAM_RESET_RELEASE_DELAY_OVERRIDE : RESET_RELEASE_DELAY_PRE;    // how many clocks late the fifo exits from safe state
    
    // reset release delay for the various occupancy trackers
    localparam int RESET_DELAY_OCC_GTE3     = RESET_RELEASE_DELAY - EARLY_MODE;
    localparam int RESET_DELAY_STALL_OUT    = RESET_RELEASE_DELAY - EARLY_MODE;
    localparam int RESET_DELAY_ALMOST_FULL  = RESET_RELEASE_DELAY;
    localparam int RESET_DELAY_ALMOST_EMPTY = RESET_RELEASE_DELAY;
    localparam int RESET_DELAY_MAX          = RESET_RELEASE_DELAY;
    
    // properties of the fifo which are consumed by the testbench
    localparam int WRITE_TO_READ_LATENCY            = 1;    //once something is written into the fifo, how many clocks later will it be visible on the read side
    localparam int RESET_EXT_HELD_LENGTH            = 4;    //if RESET_EXTERNALLY_HELD = 1, how many clocks does reset need to be held for
    localparam int MAX_CLOCKS_TO_ENTER_SAFE_STATE   = 2;    //upon assertion of reset, worse case number of clocks until fifo shows both full and empty
    localparam int MAX_CLOCKS_TO_EXIT_SAFE_STATE    = 18;   //upon release of reset, worse case number of clocks until fifo is ready to transact (not necessarily observable if INITIAL_OCCUPANCY = DEPTH)
    
    
    
    ///////////////////////////
    //                       //
    //  Signal declarations  //
    //                       //
    ///////////////////////////
    
    // Naming convention: some signals are retimed early, any signal ending with _E has the same earliness as the EARLY_MODE parameter.
    
    //reset
    genvar g;
    logic aclrn, sclrn;                             //these are the typical active low reset signals that are consumed
    logic sclrn_early_two, sclrn_early, sclrn_late; //helpers for sclrn
    logic [RESET_DELAY_MAX:0] resetn_delayed;       //delayed versions of aclrn or sclrn, consumed by the occupancy trackers
    logic fifo_in_reset;                            //intended primarily for consumption by testbench to know when fifo is in reset, also used for stall_out when NEVER_OVERFLOWS=1
    logic aclrn_occ_tracker, sclrn_occ_tracker;     //for reproducing the same reset structure that would be inside acl_tessellated_incr_decr_threshold
    
    //retime stall_in and valid_in to the correct timing, absorb excess earliness that the fifo cannot take advantage of
    logic stall_in_E, valid_in_E;
    logic [EXCESS_EARLY_STALL:0] stall_in_pipe;
    logic [EXCESS_EARLY_VALID:0] valid_in_pipe;
    
    //write control
    logic write_into_fifo, write_into_fifo_E;           //are we writing into the fifo
    logic try_write_into_fifo, try_write_into_fifo_E;   //are we trying to write into fifo, not necessarily will write, the purpose of this is to shrink the logic cone of threshold_reached in occ trackers
    logic advance_write_addr, advance_write_addr_E;     //should the write address advance
    logic ram_wr_en, ram_wr_en_E;                       //write enable to the m20k or mlab
    logic [ADDR-1:0] ram_wr_addr;                       //write address to the m20k or mlab
    
    //read control
    logic read_from_fifo, read_from_fifo_E;             //are we reading from the fifo
    logic try_read_from_fifo, try_read_from_fifo_E;     //are we trying to read from fifo, not necessarily will read
    logic advance_read_addr_E;                          //should the read address advance...
    logic m20k_addr_b_clock_en, lfsr_addr_b_incr;       //...normally we should also propagate this to the m20k read address and lfsr read address, there is a special scenario during reset exit, see comments below
    logic [ADDR-1:0] ram_rd_addr;                       //read address to the m20k or mlab
    
    //emptiness tracking - many cases to consider for how to use the data bypass for low latency
    logic occ_gte_one_E, occ_gte_two_E, occ_gte_three_E;    //used_words >= 1,2,3
    logic occ_gte_reset_exit_n, occ_gte_one_reset_exit_n, occ_gte_two_reset_exit_n; //occ_gte_three_E uses occupancy tracking, use occ encoding from acl_low_latency_fifo for others, need the re-create same reset logic
    logic valid_out_E, stall_out_E;                     //early valid out and early stall out
    
    //data bypass
    logic [WIDTH-1:0] ram_data_out;                     //read data output from m20k or mlab, can be registered or unregistered or some mix of the two depending on REGISTERED_DATA_OUT_COUNT
    logic [WIDTH-1:0] data_in_prev;                     //data_in delayed by 1 clock cycle, used exactly as in the diagram in the comments at the top
    logic [WIDTH-1:0] data_in_mux;                      //for unregistered data_out, shift some of the muxing earlier, used exactly as in the diagram in the comments at the top
    logic [WIDTH-1:0] data_out_reg, data_out_unreg;     //we compute data_out for registered and unregistered version, then data_out is just a bit select from these based on REGISTERED_DATA_OUT_COUNT
    
    //control signals for data bypass
    logic data_out_clock_en, data_out_clock_en_E;       //if data_out is registered this is the clock enable for it, otherwise this is the clock enable for the read data register inside the m20k/mlab itself
    logic sel_data_in;                                  //data_out should load from data_in when the fifo is empty
    logic sel_ram_data_out;                             //if data was written long enough ago and is now readable from the ram, use that data
    logic sel_new_data, sel_new_data_E;                 //use data_in_prev when the fifo is not empty but data is not yet readable from the ram
    
    
    
    /////////////
    //         //
    //  Reset  //
    //         //
    /////////////
    
    // the reset structure is identical to acl_mid_speed_fifo
    
    // S10 reset specification:
    // S (clocks to enter reset safe state) : 2 for sclrn_early_two to actual register (beware synchronizer takes no time for reset assertion, but it does take time for reset release)
    // P (minimum duration of reset pulse)  : 4 if RESET_EXTERNALLY_HELD = 1, otherwise 1 (we will internally pulse stretch the reset to 4 clocks)
    // D (clocks to exit reset safe state)  : 18 (3 for synchronizer) + (5 for sclrn_early_two to actual register) + (10 for reset release delay for registers that absorb excess earliness)
    
    acl_reset_handler
    #(
        .ASYNC_RESET            (ASYNC_RESET),
        .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
        .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
        .PIPE_DEPTH             (0),
        .NUM_COPIES             (1)
    )
    acl_reset_handler_inst
    (
        .clk                    (clock),
        .i_resetn               (resetn),
        .o_aclrn                (aclrn),
        .o_resetn_synchronized  (),
        .o_sclrn                (sclrn_early_two)
    );
    
    generate
    if (ASYNC_RESET) begin : async_reset
        assign sclrn = 1'b1;
        assign sclrn_late = 1'b1;
        assign sclrn_early = 1'b1;
    end
    else begin : sync_reset
        logic [2:0] sclrn_chain;
        always_ff @(posedge clock) begin
            sclrn_chain <= (sclrn_chain << 1) | sclrn_early_two;
            sclrn_early <= (RESET_EXTERNALLY_HELD) ? sclrn_early_two : ((&sclrn_chain) & sclrn_early_two);
            sclrn <= sclrn_early;
            sclrn_late <= sclrn;    //only the read address increment consumes this when using m20k and sclr
        end
    end
    endgenerate
    
    generate
    always_comb begin
        resetn_delayed[0] = (ASYNC_RESET) ? aclrn : sclrn;      //delay 0 = original reset timing
    end
    for (g=1; g<=RESET_DELAY_MAX; g++) begin : gen_resetn_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) resetn_delayed[g] <= '0;
            else begin
                resetn_delayed[g] <= (ASYNC_RESET && g==1) ? 1'b1 : resetn_delayed[g-1];    //partial reconfig requires no d-input consumption of aclr, so resetn_delayed[1] loads 1'b1 if ASYNC_RESET
                if (~sclrn_early) resetn_delayed[g] <= '0;                                  //resetn_delayed goes into reset as the same time as sclrn, since this is registered need to peek one clock ahead of sclrn
            end
        end
    end
    endgenerate
    
    //this signal is consumed by the testbench to know whether the fifo is still in reset, can't use stall_out when fifo starts as full
    //this signal may be exported by the fifo in certain configurations, e.g. NEVER_OVERFLOWS=1
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) fifo_in_reset <= 1'b1;
        else begin
            fifo_in_reset <= 1'b0;
            if (~resetn_delayed[RESET_RELEASE_DELAY]) fifo_in_reset <= 1'b1;
        end
    end
    
    
    
    ////////////////////////////////////////////////
    //                                            //
    //  Absorb excess earliness on input signals  //
    //                                            //
    ////////////////////////////////////////////////
    
    generate
    always_comb begin
        stall_in_pipe[0] = stall_in;
        valid_in_pipe[0] = valid_in;
    end
    for (g=1; g<=EXCESS_EARLY_STALL; g++) begin : gen_stall_in_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) stall_in_pipe[g] <= 1'b1;
            else begin
                stall_in_pipe[g] <= stall_in_pipe[g-1];
                if (~sclrn && RESET_EVERYTHING) stall_in_pipe[g] <= 1'b1;
            end
        end
    end
    for (g=1; g<=EXCESS_EARLY_VALID; g++) begin : gen_valid_in_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) valid_in_pipe[g] <= 1'b0;
            else begin
                valid_in_pipe[g] <= valid_in_pipe[g-1];
                if (~sclrn && RESET_EVERYTHING) valid_in_pipe[g] <= 1'b0;
            end
        end
    end
    endgenerate
    assign stall_in_E = stall_in_pipe[EXCESS_EARLY_STALL];
    assign valid_in_E = valid_in_pipe[EXCESS_EARLY_VALID];
    
    
    
    ////////////////////
    //                //
    //  Memory block  //
    //                //
    ////////////////////
    
    // We can use a mix of registered and unregistered output data, see comments at the top for how this is handled.
    // Just like acl_mid_speed_fifo, we have different implementations for M20K and MLAB. Comments from acl_mid_speed_fifo about this:
    
    // Usage of altdpram - unlike the M20K in which it is impossible to bypass the input registers (addresses, write data, write enable), for the MLAB it is possible to bypass the input
    // register for the read address. There is no parameterization of altera_syncram that supports this, hence the use of altdpram.
    
    // It is desirable to have access to the output of read address address. In the case of MLAB, the read address is driven by ALM registers. For M20K, we have no visibility on the output
    // of the read address register because this is a hardened register inside the M20K itself. For M20K we have our own read address in ALM registers which is always 1 ahead of the hardened
    // read address inside the M20K. Only when we update our read address, we assert the clock enable for the hardened read address inside the M20K, this way it always captures 1 value behind
    // what our ALM register read address is. For asynchronous reset, the M20K read address is aclr to 0 and our address starts at 1 (actually INITIAL_OCCPUPANCY=1 on the LFSR). For synchronous
    // reset, the M20K read address clock enable is active during reset so that we can clock in the value of our ALM register read address, upon reset exit the M20K clock enable is shut off
    // and our read address advances 1 step forward.
    
    assign ecc_err_status = 2'h0;   // ECC IS NOT IMPLEMENTED YET, see case:555783
    
    generate
    if (USE_MLAB) begin : gen_mlab
        if (REG_DATA_OUT_COUNT > 0) begin : gen_mlab_reg
            altdpram #(     //modelsim library: altera_mf
                .indata_aclr ("OFF"),
                .indata_reg ("INCLOCK"),
                .intended_device_family ("Stratix 10"), //quartus will correct this automatically to whatever your project actually uses
                .lpm_type ("altdpram"),
                .ram_block_type ("MLAB"),
                .outdata_aclr ("OFF"),
                .outdata_sclr ("OFF"),
                .outdata_reg ("UNREGISTERED"),          //in order to register data_out, we have to unregister the output from the ram, that way we can sneak in a 3:1 mux before data_out
                .rdaddress_aclr ("OFF"),
                .rdaddress_reg ("UNREGISTERED"),        //we own the read address, bypass the equivalent of the internal address_b from m20k
                .rdcontrol_aclr ("OFF"),
                .rdcontrol_reg ("UNREGISTERED"),
                .read_during_write_mode_mixed_ports ("DONT_CARE"),
                .width (REG_DATA_OUT_COUNT),
                .widthad (ADDR),
                .width_byteena (1),
                .wraddress_aclr ("OFF"),
                .wraddress_reg ("INCLOCK"),
                .wrcontrol_aclr ("OFF"),
                .wrcontrol_reg ("INCLOCK")
            )
            altdpram_component
            (
                //clock, no reset
                .inclock    (clock),
                .outclock   (clock),
                
                //write port
                .data       (data_in[REG_DATA_OUT_COUNT-1:0]),      //the lower bits of data_out are registered
                .wren       (ram_wr_en),
                .wraddress  (ram_wr_addr),
                
                //read port
                .rdaddress  (ram_rd_addr),
                .outclocken (1'b1),                                 //no effect since q is unregistered
                .q          (ram_data_out[REG_DATA_OUT_COUNT-1:0]), //the lower bits of data_out are registered
                
                //unused
                .aclr (1'b0),
                .sclr (1'b0),
                .byteena (1'b1),
                .inclocken (1'b1),
                .rdaddressstall (1'b0),
                .rden (1'b1),
                .wraddressstall (1'b0)
            );
        end
        if (UNREG_DATA_OUT_COUNT > 0) begin : gen_mlab_unreg
            altdpram #(     //modelsim library: altera_mf
                .indata_aclr ("OFF"),
                .indata_reg ("INCLOCK"),
                .intended_device_family ("Stratix 10"), //quartus will correct this automatically to whatever your project actually uses
                .lpm_type ("altdpram"),
                .ram_block_type ("MLAB"),
                .outdata_aclr ("OFF"),
                .outdata_sclr ("OFF"),
                .outdata_reg ("OUTCLOCK"),              //when output data is unregistered, all sources feeding that mux must be registered, one of those sources is the output of the ram
                .rdaddress_aclr ("OFF"),
                .rdaddress_reg ("UNREGISTERED"),        //we own the read address, bypass the equivalent of the internal address_b from m20k
                .rdcontrol_aclr ("OFF"),
                .rdcontrol_reg ("UNREGISTERED"),
                .read_during_write_mode_mixed_ports ("DONT_CARE"),
                .width (UNREG_DATA_OUT_COUNT),
                .widthad (ADDR),
                .width_byteena (1),
                .wraddress_aclr ("OFF"),
                .wraddress_reg ("INCLOCK"),
                .wrcontrol_aclr ("OFF"),
                .wrcontrol_reg ("INCLOCK")
            )
            altdpram_component
            (
                //clock, no reset
                .inclock    (clock),
                .outclock   (clock),
                
                //write port
                .data       (data_in[WIDTH-1:REG_DATA_OUT_COUNT]),      //the upper bits of data_out are unregistered
                .wren       (ram_wr_en),
                .wraddress  (ram_wr_addr),
                
                //read port
                .rdaddress  (ram_rd_addr),
                .outclocken (data_out_clock_en),                        //q is registered and we need access to its clock enable
                .q          (ram_data_out[WIDTH-1:REG_DATA_OUT_COUNT]), //the upper bits of data_out are unregistered
                
                //unused
                .aclr (1'b0),
                .sclr (1'b0),
                .byteena (1'b1),
                .inclocken (1'b1),
                .rdaddressstall (1'b0),
                .rden (1'b1),
                .wraddressstall (1'b0)
            );
        end
    end
    else begin : gen_m20k
        if (REG_DATA_OUT_COUNT > 0) begin : gen_m20k_reg
            altera_syncram #(   //modelsim library: altera_lnsim
                .numwords_a (2**ADDR),
                .numwords_b (2**ADDR),
                .address_aclr_b ((ASYNC_RESET) ? "CLEAR1" : "NONE"),
                .address_reg_b ("CLOCK1"),
                .clock_enable_input_a ("BYPASS"),
                .clock_enable_input_b ("BYPASS"),
                .clock_enable_output_b ("BYPASS"),      //q is unregistered, so clock enable has no effect
                .enable_ecc ("FALSE"),
                .intended_device_family ("Stratix 10"), //quartus will correct this automatically to whatever your project actually uses
                .lpm_type ("altera_syncram"),
                .operation_mode ("DUAL_PORT"),
                .outdata_aclr_b ("NONE"),
                .outdata_sclr_b ("NONE"),
                .outdata_reg_b ("UNREGISTERED"),        //in order to register data_out, we have to unregister the output from the ram, that way we can sneak in a 3:1 mux before data_out
                .power_up_uninitialized ("TRUE"),
                .ram_block_type ("M20K"),
                .read_during_write_mode_mixed_ports ("DONT_CARE"),
                .widthad_a (ADDR),
                .widthad_b (ADDR),
                .width_a (REG_DATA_OUT_COUNT),
                .width_b (REG_DATA_OUT_COUNT),
                .width_byteena_a (1)
            )
            altera_syncram
            (
                //clock and reset
                .clock0         (clock),
                .clock1         (clock),
                .aclr1          ((ASYNC_RESET) ? ~aclrn : 1'b0),        //this is used to reset the internal address_b when ASYNC_RESET=1
                
                //write port
                .wren_a         (ram_wr_en),
                .address_a      (ram_wr_addr),
                .data_a         (data_in[REG_DATA_OUT_COUNT-1:0]),      //the lower bits of data_out are registered
                
                //read port
                .address_b      (ram_rd_addr),
                .addressstall_b (~m20k_addr_b_clock_en),
                .clocken1       (1'b1),                                 //no effect since q is unregistered
                .q_b            (ram_data_out[REG_DATA_OUT_COUNT-1:0]), //the lower bits of data_out are registered
                
                //unused
                .aclr0 (1'b0),
                .address2_a (1'b1),
                .address2_b (1'b1),
                .addressstall_a (1'b0),
                .byteena_a (1'b1),
                .byteena_b (1'b1),
                .clocken0 (1'b1),
                .clocken2 (1'b1),
                .clocken3 (1'b1),
                .data_b ({WIDTH{1'b1}}),
                .eccencbypass (1'b0),
                .eccencparity (8'b0),
                .eccstatus (),
                .q_a (),
                .rden_a (1'b1),
                .rden_b (1'b1),
                .sclr (1'b0),
                .wren_b (1'b0)
            );
        end
        if (UNREG_DATA_OUT_COUNT > 0) begin : gen_m20k_unreg
            altera_syncram #(   //modelsim library: altera_lnsim
                .numwords_a (2**ADDR),
                .numwords_b (2**ADDR),
                .address_aclr_b ((ASYNC_RESET) ? "CLEAR1" : "NONE"),
                .address_reg_b ("CLOCK1"),
                .clock_enable_input_a ("BYPASS"),
                .clock_enable_input_b ("BYPASS"),
                .clock_enable_output_b ("NORMAL"),      //clock enable for output data register is controlled by clocken1
                .enable_ecc ("FALSE"),
                .intended_device_family ("Stratix 10"), //quartus will correct this automatically to whatever your project actually uses
                .lpm_type ("altera_syncram"),
                .operation_mode ("DUAL_PORT"),
                .outdata_aclr_b ("NONE"),
                .outdata_sclr_b ("NONE"),
                .outdata_reg_b ("CLOCK1"),              //when output data is unregistered, all sources feeding that mux must be registered, one of those sources is the output of the ram
                .power_up_uninitialized ("TRUE"),
                .ram_block_type ("M20K"),
                .read_during_write_mode_mixed_ports ("DONT_CARE"),
                .widthad_a (ADDR),
                .widthad_b (ADDR),
                .width_a (UNREG_DATA_OUT_COUNT),
                .width_b (UNREG_DATA_OUT_COUNT),
                .width_byteena_a (1)
            )
            altera_syncram
            (
                //clock and reset
                .clock0         (clock),
                .clock1         (clock),
                .aclr1          ((ASYNC_RESET) ? ~aclrn : 1'b0),    //this is used to reset the internal address_b when ASYNC_RESET=1
                
                //write port
                .wren_a         (ram_wr_en),
                .address_a      (ram_wr_addr),
                .data_a         (data_in[WIDTH-1:REG_DATA_OUT_COUNT]),      //the upper bits of data_out are unregistered
                
                //read port
                .address_b      (ram_rd_addr),
                .addressstall_b (~m20k_addr_b_clock_en),
                .clocken1       (data_out_clock_en),                        //q is registered and we need access to its clock enable
                .q_b            (ram_data_out[WIDTH-1:REG_DATA_OUT_COUNT]), //the upper bits of data_out are unregistered
                
                //unused
                .aclr0 (1'b0),
                .address2_a (1'b1),
                .address2_b (1'b1),
                .addressstall_a (1'b0),
                .byteena_a (1'b1),
                .byteena_b (1'b1),
                .clocken0 (1'b1),
                .clocken2 (1'b1),
                .clocken3 (1'b1),
                .data_b ({WIDTH{1'b1}}),
                .eccencbypass (1'b0),
                .eccencparity (8'b0),
                .eccstatus (),
                .q_a (),
                .rden_a (1'b1),
                .rden_b (1'b1),
                .sclr (1'b0),
                .wren_b (1'b0)
            );
        end
    end
    endgenerate
    
    
    
    /////////////////////
    //                 //
    //  Write address  //
    //                 //
    /////////////////////
    
    assign advance_write_addr_E = ~stall_out_E & valid_in_E & (occ_gte_two_E | (valid_out_E & stall_in_E)); //don't write to fifo if data captured persistently in the data bypass (data_out if registered, otherwise data_in_mux)
    assign ram_wr_en_E = ~stall_out_E & valid_in_E;     //simplify the write logic, this allows more writes than absolutely necessary in order to ease routing to the write enable of all the physical memories
    
    assign try_write_into_fifo_E = valid_in_E;
    assign write_into_fifo_E = valid_in_E & ~stall_out_E;
    
    generate
    if (EARLY_MODE == 0) begin : write_incr0
        assign ram_wr_en = ram_wr_en_E;
        assign advance_write_addr = advance_write_addr_E;
        assign try_write_into_fifo = try_write_into_fifo_E;
        assign write_into_fifo = write_into_fifo_E;
    end
    if (EARLY_MODE == 1) begin : write_incr1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                ram_wr_en <= 1'b0;
                advance_write_addr <= 1'b0;
                write_into_fifo <= 1'b0;
            end
            else begin
                ram_wr_en <= ram_wr_en_E;
                advance_write_addr <= advance_write_addr_E;
                write_into_fifo <= write_into_fifo_E;
                if (~sclrn) begin
                    ram_wr_en <= 1'b0;
                    advance_write_addr <= 1'b0;
                    write_into_fifo <= 1'b0;
                end
            end
        end
        assign try_write_into_fifo = write_into_fifo;
    end
    endgenerate
    
    acl_lfsr #(
        .WIDTH                  (ADDR),
        .ASYNC_RESET            (ASYNC_RESET),
        .SYNCHRONIZE_RESET      (0),
        .INITIAL_OCCUPANCY      ((INITIAL_OCCUPANCY > 1) ? (INITIAL_OCCUPANCY - 1) : 0)     //we don't advance the write address when fifo is empty, it gets captured in bypass register
    )
    m20k_wraddr_inst
    (
        .clock                  (clock),
        .resetn                 (resetn_delayed[0]),
        .enable                 (advance_write_addr),
        .state                  (ram_wr_addr)
    );
    
    
    
    ////////////////////
    //                //
    //  Read address  //
    //                //
    ////////////////////
    
    // Like acl_mid_speed_fifo, for M20K with synchronous reset we have special logic for clocking in our ALM register read address into the hardened read address inside the M20K itself. See comments there for how this works.
    
    assign advance_read_addr_E = occ_gte_two_E & ~stall_in_E;   //if we didn't write incoming data into the ram, also don't try to read it from the ram
    
    generate
    if (EARLY_MODE == 0) begin : read_incr0
        assign lfsr_addr_b_incr     = (!USE_MLAB && !ASYNC_RESET) ? (advance_read_addr_E | ~sclrn_late) : advance_read_addr_E;
        assign m20k_addr_b_clock_en = (!USE_MLAB && !ASYNC_RESET) ? (advance_read_addr_E | ~sclrn)      : advance_read_addr_E;
    end
    if (EARLY_MODE == 1) begin : read_incr1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                lfsr_addr_b_incr <= 1'b0;
                m20k_addr_b_clock_en <= 1'b0;
            end
            else begin
                lfsr_addr_b_incr <= advance_read_addr_E;
                m20k_addr_b_clock_en <= advance_read_addr_E;
                if (!USE_MLAB && !ASYNC_RESET) begin    //special reset behavior for M20K using sync reset, peek one stage ahead on sclr since we are now registering these signals
                    if (~sclrn) lfsr_addr_b_incr <= 1'b1;
                    if (~sclrn_early) m20k_addr_b_clock_en <= 1'b1;
                end
                else begin                              //normal reset behavior
                    if (~sclrn && RESET_EVERYTHING) begin
                        lfsr_addr_b_incr <= 1'b0;
                        m20k_addr_b_clock_en <= 1'b0;
                    end
                end
            end
        end
    end
    endgenerate
    
    acl_lfsr #(
        .WIDTH                  (ADDR),
        .ASYNC_RESET            (ASYNC_RESET),
        .SYNCHRONIZE_RESET      (0),
        .INITIAL_OCCUPANCY      ((!USE_MLAB && ASYNC_RESET) ? 1 : 0)    //for M20K our read address is 1 ahead of the address_b inside the M20K, for async reset we don't clock our address during reset, so just start 1 ahead
    )
    m20k_rdaddr_inst
    (
        .clock                  (clock),
        .resetn                 (resetn_delayed[0]),
        .enable                 (lfsr_addr_b_incr),
        .state                  (ram_rd_addr)
    );
    
    
    
    //////////////////////////
    //                      //
    //  Emptiness tracking  //
    //                      //
    //////////////////////////
    
    // occ_gte_X checks whether used_words >= X. For a latency one fifo, write_used_words == read_used_words. We use the same encoding of occ from acl_low_latency_fifo:
    //
    // Let occ[2:0] = {occ_gte_three_E, occ_gte_two_E, occ_gte_one_E}
    //
    // value of occ | actual occupancy
    // -------------+-----------------
    // 3'b000       | 0
    // 3'b001       | 1
    // 3'b011       | 2
    // 3'b111       | 3+
    //
    // Only occ_gte_three_E uses an occupancy tracker, the pass-it-along-from-your-neighbor approach from acl_low_latency_fifo is used to generate occ_gte_two_E and occ_gte_one_E. Special care is needed for shallow
    // fifos where things get a bit complicated with INITIAL_OCCUPANCY and WRITE_AND_READ_DURING_FULL. Ideally all 3 occ_gte_X signals should share the same reset, but since occ_gte_three_E is driven from an
    // occupancy tracker, we need to re-create the same reset control that the occupancy tracker would have used, which is commonized in occ_gte_reset_exit_n, and depending on INITIAL_OCCUPANCY we adjust the actual
    // reset that occ_gte_two_E and occ_gte_one_E consume.
    
    //mimic the reset behavior inside acl_tessellated_incr_decr_threshold -- in case THRESHOLD_REACHED_AT_RESET is different than the ideal value, we need a way to restore the ideal value at reset exit
    assign aclrn_occ_tracker =  (ASYNC_RESET) ? resetn_delayed[RESET_DELAY_OCC_GTE3] : 1'b1;
    assign sclrn_occ_tracker = (!ASYNC_RESET) ? resetn_delayed[RESET_DELAY_OCC_GTE3] : 1'b1;
    always_ff @(posedge clock or negedge aclrn_occ_tracker) begin
        if (~aclrn_occ_tracker) occ_gte_reset_exit_n <= 1'b0;
        else occ_gte_reset_exit_n <= sclrn_occ_tracker;
    end
    assign occ_gte_one_reset_exit_n = (INITIAL_OCCUPANCY == 0) ? 1'b1 : occ_gte_reset_exit_n;
    assign occ_gte_two_reset_exit_n = (INITIAL_OCCUPANCY <= 1) ? 1'b1 : occ_gte_reset_exit_n;
    
    always_ff @(posedge clock or negedge aclrn_occ_tracker) begin
        if (~aclrn_occ_tracker) occ_gte_one_E <= 1'b0;
        else begin
            if (DEPTH == 1) begin
                if (~stall_in_E & occ_gte_one_E & ((WRITE_AND_READ_DURING_FULL) ? ~valid_in_E : 1'b1)) occ_gte_one_E <= 1'b0;
                else if (valid_in_E & ~stall_out_E) occ_gte_one_E <= 1'b1;
            end
            else begin
                if (valid_in_E & ((INITIAL_OCCUPANCY == 0) ? ~stall_out_E : 1'b1)) occ_gte_one_E <= 1'b1;
                else if (~stall_in_E & ~occ_gte_two_E) occ_gte_one_E <= 1'b0;
            end
            if (~occ_gte_one_reset_exit_n) occ_gte_one_E <= 1'b1;
            if (~sclrn_occ_tracker) occ_gte_one_E <= 1'b0;
        end
    end
    
    generate
    if (DEPTH >= 2) begin : gen_occ_gte_two_E
        always_ff @(posedge clock or negedge aclrn_occ_tracker) begin
            if (~aclrn_occ_tracker) occ_gte_two_E <= 1'b0;
            else begin
                if (valid_in_E & stall_in_E & occ_gte_one_E) occ_gte_two_E <= 1'b1;
                else if (~stall_in_E & ~occ_gte_three_E & (((DEPTH==2) && (WRITE_AND_READ_DURING_FULL==0)) ? 1'b1 : ~valid_in_E)) occ_gte_two_E <= 1'b0;
                if (~occ_gte_two_reset_exit_n) occ_gte_two_E <= 1'b1;
                if (~sclrn_occ_tracker) occ_gte_two_E <= 1'b0;
            end
        end
    end
    else begin : zero_occ_gte_two_E
        assign occ_gte_two_E = 1'b0;
    end
    endgenerate
    
    generate
    if (DEPTH >= 3) begin : gen_occ_gte_three_E
        localparam bit OCC_GTE3_GUARD_INCR_RAW = (INITIAL_OCCUPANCY == 2) ? 1'b1 : 1'b0;    //same problem as almost_full but in the opposite direction (valid_in=1 during reset)
        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DEPTH),
            .THRESHOLD                  (3),
            .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
            .THRESHOLD_REACHED_AT_RESET (0),
            .WRITE_AND_READ_DURING_FULL (WRITE_AND_READ_DURING_FULL),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0),
            .RESET_EVERYTHING           (RESET_EVERYTHING),
            .RESET_EXTERNALLY_HELD      (1)
        )
        occ_gte_three_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_delayed[RESET_DELAY_OCC_GTE3]),
            .incr_no_overflow           (write_into_fifo_E),
            .incr_raw                   ((OCC_GTE3_GUARD_INCR_RAW) ? write_into_fifo_E : try_write_into_fifo_E),
            .decr_no_underflow          (read_from_fifo_E),
            .decr_raw                   (try_read_from_fifo_E),
            .threshold_reached          (occ_gte_three_E)
        );
    end
    else begin : zero_occ_gte_three_E
        assign occ_gte_three_E = 1'b0;
    end
    endgenerate
    
    //read logic
    assign valid_out_E = occ_gte_one_E;
    assign try_read_from_fifo_E = ~stall_in_E;
    assign read_from_fifo_E = valid_out_E & ~stall_in_E;
    assign forced_read_out = read_from_fifo;
    
    //correct timing to no earliness
    generate
    if (EARLY_MODE == 0) begin : read0
        assign valid_out = occ_gte_one_E;
        assign try_read_from_fifo = try_read_from_fifo_E;
        assign read_from_fifo = read_from_fifo_E;
    end
    if (EARLY_MODE == 1) begin : read1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                valid_out <= 1'b0;
                read_from_fifo <= 1'b0;
            end
            else begin
                valid_out <= occ_gte_one_E;
                read_from_fifo <= read_from_fifo_E;
                if (~sclrn) begin
                    valid_out <= 1'b0;
                    read_from_fifo <= 1'b0;
                end
            end
        end
        assign try_read_from_fifo = read_from_fifo;
    end
    endgenerate
    
    
    
    ///////////////////////////////////
    //                               //
    //  Data bypass for low latency  //
    //                               //
    ///////////////////////////////////
    
    //data_out_clock_en is the clock enable for data_out when registered, otherwise it is clock enable for the read data register inside the M20K or MLAB itself
    //note that both scenarios can be used at the same time if REGISTERED_DATA_OUT_COUNT results in some bits of data_out being registered and some being unregistered
    //normally data_out should load if the fifo is empty or downstream says it can accept data, things get complicated when we must hold the last value when the fifo empties
    generate
    if (!HOLD_DATA_OUT_WHEN_EMPTY) begin : simple_data_out_clock_en_E
        assign data_out_clock_en_E = ~occ_gte_one_E | ~stall_in_E;
    end
    else begin : hold_data_out_clock_en_E
        logic empty_and_writing_E, one_item_and_writing_and_reading_E, two_or_more_items_and_reading_E;
        assign empty_and_writing_E = ~occ_gte_one_E & valid_in_E;
        assign one_item_and_writing_and_reading_E = ((DEPTH==1) && (WRITE_AND_READ_DURING_FULL==0)) ? 1'b0 : (occ_gte_one_E & ~occ_gte_two_E & ~stall_in_E & valid_in_E);
        assign two_or_more_items_and_reading_E = occ_gte_two_E & ~stall_in_E;
        assign data_out_clock_en_E = empty_and_writing_E | one_item_and_writing_and_reading_E | two_or_more_items_and_reading_E;
    end
    endgenerate
    
    //sel_data_in means data_out should source from data_in, which happens when the fifo is empty, or there is 1 word and we are reading and writing
    //sel_new_data means data_out should source from data_in_prev -- this is essentially the new data mode discussed in the comments at the top which is used to lower the write to latency from 3 to 2
    //in all other scenarios, data_out should source from the ram read data
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) sel_new_data_E <= 1'b0;
        else begin
            sel_new_data_E <= (stall_in_E) ? (occ_gte_one_E & ~occ_gte_two_E) : (occ_gte_two_E & ~occ_gte_three_E);
            if (~sclrn && RESET_EVERYTHING) sel_new_data_E <= 1'b0;
        end
    end
    
    //correct timing to no earliness -- control signal operate with earliness but the data path itself does not, this gives time for the control signals to fan-out to the entire data path width
    generate
    if (EARLY_MODE == 0) begin : data0
        assign data_out_clock_en = data_out_clock_en_E;
        assign sel_new_data = sel_new_data_E;
        assign sel_data_in = ~occ_gte_two_E;
    end
    if (EARLY_MODE == 1) begin : data1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_out_clock_en <= 1'b0;
                sel_new_data <= 1'b0;
                sel_data_in <= 1'b0;
            end
            else begin
                data_out_clock_en <= data_out_clock_en_E;
                sel_new_data <= sel_new_data_E;
                sel_data_in <= ~occ_gte_two_E;
                if (~sclrn && RESET_EVERYTHING) begin
                    data_out_clock_en <= 1'b0;
                    sel_new_data <= 1'b0;
                    sel_data_in <= 1'b0;
                end
            end
        end
    end
    endgenerate
    
    //data path for both registered and unregistered data_out
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            data_in_prev <= '0;
            data_out_reg <= '0;
            data_in_mux <= '0;
            sel_ram_data_out <= 1'b0;
        end
        else begin
            data_in_prev <= data_in;
            if (data_out_clock_en) begin
                data_out_reg <= (sel_data_in) ? data_in : (sel_new_data) ? data_in_prev : ram_data_out;
                data_in_mux <= (sel_data_in) ? data_in : data_in_prev;
                sel_ram_data_out <= ~sel_data_in & ~sel_new_data;
            end
            if (~sclrn && RESET_EVERYTHING) begin
                data_in_prev <= '0;
                data_out_reg <= '0;
                data_in_mux <= '0;
                sel_ram_data_out <= 1'b0;
            end
        end
    end
    assign data_out_unreg = (sel_ram_data_out) ? ram_data_out : data_in_mux;
    
    //select whether to use registered or unregistered data_out
    generate
    if (REG_DATA_OUT_COUNT == 0) begin : gen_unregistered_data_out
        assign data_out = data_out_unreg;
    end
    else if (REG_DATA_OUT_COUNT == WIDTH) begin : gen_registered_data_out
        assign data_out = data_out_reg;
    end
    else begin : gen_mixed_reg_data_out
        assign data_out = {data_out_unreg[WIDTH-1:REG_DATA_OUT_COUNT], data_out_reg[REG_DATA_OUT_COUNT-1:0]};
    end
    endgenerate
    
    
    
    /////////////////
    //             //
    //  Fifo full  //
    //             //
    /////////////////
    
    generate
    if (NEVER_OVERFLOWS) begin : gen_reset_stall_out    //no overflow protection, but upstream still needs a way to know when fifo has exited from reset
        if (EARLY_MODE == 0) begin : fifo_in_reset0
            assign stall_out_E = fifo_in_reset;
        end
        if (EARLY_MODE == 1) begin : fifo_in_reset1
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) stall_out_E <= 1'b1;
                else begin
                    stall_out_E <= 1'b0;
                    if (~resetn_delayed[RESET_RELEASE_DELAY-1]) stall_out_E <= 1'b1;   //RESET_RELEASE_DELAY will be at least 1 when EARLY_MODE is 1
                end
            end
        end
    end
    else begin : gen_real_stall_out
        localparam bit STALL_OUT_GUARD_DECR_RAW = (DEPTH == INITIAL_OCCUPANCY) ? 1'b1 : 1'b0;   //if fifo starts full and stall_in=0 during reset, it will cause stall_out to deassert 1 clock earlier than expected
        logic stall_out_E_raw;
        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DEPTH),
            .THRESHOLD                  (DEPTH),
            .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
            .THRESHOLD_REACHED_AT_RESET (1),
            .WRITE_AND_READ_DURING_FULL (WRITE_AND_READ_DURING_FULL),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0),
            .RESET_EVERYTHING           (RESET_EVERYTHING),
            .RESET_EXTERNALLY_HELD      (1)
        )
        stall_out_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_delayed[RESET_DELAY_STALL_OUT]),
            .incr_no_overflow           (write_into_fifo_E),
            .incr_raw                   (try_write_into_fifo_E),
            .decr_no_underflow          (read_from_fifo_E),
            .decr_raw                   ((STALL_OUT_GUARD_DECR_RAW) ? read_from_fifo_E : try_read_from_fifo_E),
            .threshold_reached          (stall_out_E_raw)
        );
        assign stall_out_E = (!WRITE_AND_READ_DURING_FULL) ? stall_out_E_raw : (stall_out_E_raw & ~read_from_fifo_E);
    end
    endgenerate
    
    //correct timing to no earliness
    generate
    if (EARLY_MODE == 0) begin : stall_out0
        assign stall_out = stall_out_E;
    end
    if (EARLY_MODE == 1) begin : stall_out1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) stall_out <= 1'b1;
            else begin
                stall_out <= stall_out_E;
                if (~sclrn) stall_out <= 1'b1;
            end
        end
    end
    endgenerate
    
    
    
    ///////////////////
    //               //
    //  Almost full  //
    //               //
    ///////////////////
    
    generate
    if ((ALMOST_FULL_CUTOFF == 0) && (NEVER_OVERFLOWS == 0) && (WRITE_AND_READ_DURING_FULL == 0)) begin : full_almost_full
        assign almost_full = stall_out;
    end
    else begin : real_almost_full
        localparam bit ALMOST_FULL_GUARD_DECR_RAW = ((DEPTH-ALMOST_FULL_CUTOFF) == INITIAL_OCCUPANCY) ? 1'b1 : 1'b0;    //same problem as stall_out, if stall_in=0 during reset then almost_full deasserts too early
        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DEPTH),
            .THRESHOLD                  (DEPTH - ALMOST_FULL_CUTOFF),
            .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
            .THRESHOLD_REACHED_AT_RESET (1),
            .WRITE_AND_READ_DURING_FULL (WRITE_AND_READ_DURING_FULL),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0),
            .RESET_EVERYTHING           (RESET_EVERYTHING),
            .RESET_EXTERNALLY_HELD      (1)
        )
        almost_full_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_delayed[RESET_DELAY_ALMOST_FULL]),
            .incr_no_overflow           (write_into_fifo),
            .incr_raw                   (try_write_into_fifo),
            .decr_no_underflow          (read_from_fifo),
            .decr_raw                   ((ALMOST_FULL_GUARD_DECR_RAW) ? read_from_fifo : try_read_from_fifo),
            .threshold_reached          (almost_full)
        );
    end
    endgenerate
    
    
    
    ////////////////////
    //                //
    //  Almost empty  //
    //                //
    ////////////////////
    
    generate
    if (ALMOST_EMPTY_CUTOFF == 0) begin : empty_almost_empty
        assign almost_empty = ~valid_out;
    end
    else if (ALMOST_EMPTY_CUTOFF <= 2) begin : small_almost_empty   //reuse the occupancy information we already have available
        if (EARLY_MODE == 0) begin
            assign almost_empty = (ALMOST_EMPTY_CUTOFF == 1) ? ~occ_gte_two_E : ~occ_gte_three_E;
        end
        else begin
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) almost_empty <= 1'b1;
                else begin
                    almost_empty <= (ALMOST_EMPTY_CUTOFF == 1) ? ~occ_gte_two_E : ~occ_gte_three_E;
                    if (~sclrn) almost_empty <= 1'b1;
                end
            end
        end
    end
    else begin : real_almost_empty
        localparam bit ALMOST_EMPTY_GUARD_INCR_RAW = (ALMOST_EMPTY_CUTOFF == INITIAL_OCCUPANCY) ? 1'b1 : 1'b0;  //same problem as almost_full but in the opposite direction (valid_in=1 during reset)
        logic not_almost_empty;
        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DEPTH),
            .THRESHOLD                  (ALMOST_EMPTY_CUTOFF + 1),
            .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
            .THRESHOLD_REACHED_AT_RESET (0),
            .WRITE_AND_READ_DURING_FULL (WRITE_AND_READ_DURING_FULL),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0),
            .RESET_EVERYTHING           (RESET_EVERYTHING),
            .RESET_EXTERNALLY_HELD      (1)
        )
        almost_empty_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_delayed[RESET_DELAY_ALMOST_EMPTY]),
            .incr_no_overflow           (write_into_fifo),
            .incr_raw                   ((ALMOST_EMPTY_GUARD_INCR_RAW) ? write_into_fifo : try_write_into_fifo),
            .decr_no_underflow          (read_from_fifo),
            .decr_raw                   (try_read_from_fifo),
            .threshold_reached          (not_almost_empty)
        );
        assign almost_empty = ~not_almost_empty;
    end
    endgenerate
    
    
    
    //////////////////////////////////////////////////////////
    //                                                      //
    //  Export signals needed by acl_latency_zero_ram_fifo  //
    //                                                      //
    //////////////////////////////////////////////////////////
    
    assign zlram_occ_gte_one_E = occ_gte_one_E;
    assign zlram_stall_out_E = stall_out_E;
    
endmodule

`default_nettype wire
