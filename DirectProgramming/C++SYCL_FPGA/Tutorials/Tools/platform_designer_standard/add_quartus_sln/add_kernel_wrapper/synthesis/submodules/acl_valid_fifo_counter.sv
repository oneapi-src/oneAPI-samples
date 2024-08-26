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

`default_nettype none

module acl_valid_fifo_counter
#(
    parameter int DEPTH = 0,            // >0 -- capacity of the fifo or whatever you are tracking the occupancy of, instantiator must set this parameter
    parameter int STRICT_DEPTH = 0,     // 0|1 -- using this when DEPTH > 5 has an area and fmax penalty
    parameter int ALLOW_FULL_WRITE = 0, // 0|1 -- 1 means valid_in is still accepted if FIFO is full and we are currently reading from it, this has a big fmax penalty if DEPTH > 1
    parameter bit ASYNC_RESET = 1,      // how do the registers CONSUME reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET = 0 // before consumption, do we SYNCHRONIZE the reset: 1 means use a synchronizer (assume reset arrived asynchronously), 0 means passthrough (assume reset was already synchronized)
)
(
    input  wire  clock,
    input  wire  resetn,                // if ASYNC_RESET = 0, then reset must be held for at least 4 clocks to let it propagate through all control registers
    input  wire  valid_in,              // upstream advertises it has data available
    output logic valid_out,             // we advertise to downstream there is
    input  wire  stall_in,              // asserted means downstream applies backpressure
    output logic stall_out,             // asserted means we apply backpressure to upstream
    output logic empty,                 // occupancy == 0
    output logic full                   // occupancy == DEPTH if STRICT_DEPTH=1, otherwise if full is asserted then we have occupancy >= DEPTH (having DEPTH or more items does not guarantee full will assert)
);
    //synthesis translate_off
    generate
    if (DEPTH < 1) begin
        $fatal(1, "acl_valid_fifo_counter: illegal value of DEPTH = %d, minimum allowed is 1\n", DEPTH);
    end
    endgenerate
    //synthesis translate_on

    // the functionality of this circuit is to track occupancy:
    // logic write, read;
    // logic [$clog2(DEPTH+1)-1:0] used_words;
    // assign write = valid_in & ~stall_out;
    // assign read = valid_out & ~stall_in;
    // always_ff @(posedge clock) begin
    //     used_words <= used_words + write - read;
    //     if (~resetn) used_words <= 'h0;
    // end
    // assign valid_out = (used_words > 0);
    // assign full = (used_words == DEPTH);

    logic aclrn, sclrn;
    acl_reset_handler
    #(
        .ASYNC_RESET            (ASYNC_RESET),
        .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
        .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
        .PULSE_EXTENSION        (0),
        .PIPE_DEPTH             (1),
        .NUM_COPIES             (1)
    )
    acl_reset_handler_inst
    (
        .clk                    (clock),
        .i_resetn               (resetn),
        .o_aclrn                (aclrn),
        .o_resetn_synchronized  (),
        .o_sclrn                (sclrn)
    );

    //these are somewhat redundant signals
    assign empty = ~valid_out;
    assign stall_out = (ALLOW_FULL_WRITE) ? (full & stall_in) : full;

    generate
    if (DEPTH == 1) begin : gen_depth_1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                valid_out <= 1'b0;
            end
            else begin
                if (ALLOW_FULL_WRITE) begin
                    valid_out <= (valid_in & ~valid_out) ? 1'b1 : (~stall_in & valid_out & ~valid_in) ? 1'b0 : valid_out;
                end
                else begin
                    valid_out <= (valid_in & ~valid_out) ? 1'b1 : (~stall_in & valid_out) ? 1'b0 : valid_out;
                end
                if (~sclrn) valid_out <= 1'b0;
            end
        end
        assign full = valid_out;
    end
    else if (DEPTH <= 5) begin : gen_depth_small
        logic [DEPTH-1:0] occ;  //occ[i] means the occupancy is at least i+1

        always_ff @(posedge clock or negedge aclrn) begin
        integer i;
            if (~aclrn) begin
                occ <= 'h0;
            end
            else begin
                occ[0] <= (valid_in) ? 1'b1 : (~stall_in) ? occ[1] : occ[0];

                for (i=1; i<DEPTH-1; i=i+1) begin : GEN_RANDOM_BLOCK_NAME_R66
                    occ[i] <= (valid_in & stall_in) ? occ[i-1] : (~valid_in & ~stall_in) ? occ[i+1] : occ[i];
                end

                if (ALLOW_FULL_WRITE) begin
                    occ[DEPTH-1] <= (~stall_in & ~valid_in) ? 1'b0 : (valid_in & stall_in) ? occ[DEPTH-2] : occ[DEPTH-1];
                end
                else begin
                    occ[DEPTH-1] <= (~stall_in) ? 1'b0 : (valid_in) ? occ[DEPTH-2] : occ[DEPTH-1];
                end

                if (~sclrn) occ <= 'h0;
            end
        end
        assign valid_out = occ[0];
        assign full = occ[DEPTH-1];
    end
    else begin : gen_depth_large
        // real:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        // lo:      0   1   2   3   0   1   2   3   0   1   2   3   0   1   2   3   0   1   2   3
        // aux:     0   0   x   x   1   1   x   x   0   0   x   x   1   1   x   x   0   0   x   x
        // cnt_up:  0   0   0   0  -1  +1   0   0   0   0   0   0  -1  +1   0   0   0   0   0   0
        // cnt:     0   0   x   x   x   x   x   x   1   1   x   x   x   x   x   x   2   2   x   x
        //
        // cnt updates when real % 8 changes between 4 and 5, but cnt is 3 clocks late so only when we are 3 away from this boundary
        // then cnt is deterministically known, i.e. cnt has updated by the time real % 8 has changed to 0 or 1
        //
        // cnt will be 0 or 1 when real occupancy is between 2 to 7 inclusive, real = 18 is the smallest occupancy where cnt could be 3
        // likewise real = 50 is the smallest occupancy where cnt could be 7
        //
        //  WIDTH   max DEPTH
        //  2       18 = 2**(WIDTH+3) - 14
        //  3       50 = 2**(WIDTH+3) - 14
        //  4       114 = 2**(WIDTH+3) - 14
        //  5       241 = 2**(WIDTH+3) - 15
        //  6       497 = 2**(WIDTH+3) - 15

        localparam WIDTH = (DEPTH<=18) ? 2 : (DEPTH<=50) ? 3 : (DEPTH<=114) ? 4 : ($clog2(DEPTH+15)-3);

        logic         [1:0] lo, lo_prev;
        logic               aux, flip_aux;
        logic         [1:0] cnt_up;                     //1 clock behind lo
        logic   [WIDTH-1:0] cnt;                        //2 clocks behind lo
        logic               cnt_at_target;              //3 clocks behind lo
        logic               cnt_upper_all_zeros;
        logic               cnt_all_ones;

        logic               exact_full, approx_full;    //only 1 of these will be exposed as output signal full, let quartus trim away the unused parts

        logic               inv_aux, flip_inv_aux;
        logic         [1:0] inv_cnt_up;
        logic   [WIDTH-1:0] inv_cnt;
        logic               inv_cnt_at_target;
        logic               inv_cnt_upper_all_zeros;

        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                lo <= 2'h0;
                lo_prev <= 2'h0;
                flip_aux <= 1'b0;
                aux <= 1'b0;
                flip_inv_aux <= 1'b0;
                inv_aux <= ((DEPTH+1)/4) & 1;
                cnt_up <= 2'h0;
                inv_cnt_up <= 2'h0;
                cnt <= 0;
                inv_cnt <= (DEPTH+3)/8;
                cnt_upper_all_zeros <= 1'b0;
                inv_cnt_upper_all_zeros <= 1'b0;
                cnt_all_ones <= 1'b0;
                cnt_at_target <= 1'b0;
                inv_cnt_at_target <= 1'b0;
                valid_out <= 1'b0;
                approx_full <= 1'b0;
                exact_full <= 1'b0;
            end
            else begin
                //lo tracks the bottom 2 bits of the occupancy
                lo[0] <= lo[0] ^ (valid_in & ~stall_out) ^ (valid_out & ~stall_in);
                lo[1] <= lo[1] ^ ((valid_in & ~stall_out) & ~(valid_out & ~stall_in) & lo[0]) ^ (~(valid_in & ~stall_out) & (valid_out & ~stall_in) & ~lo[0]);
                lo_prev <= lo;

                //aux tracks bit 2 of the occupancy, increment when lo goes from 2->3, decrement when lo goes from 3->2
                //since aux is 1 clock cycle late compared to lo, then aux is stable with lo % 4 == 0 and lo % 4 == 1
                //only a single bit counter, so increment and decrement both mean flip the value
                flip_aux <= ((lo==2'h2) & valid_in & ~stall_out & stall_in) | ((lo==2'h3) & ~(valid_in & ~stall_out) & ~stall_in);
                aux <= aux ^ flip_aux;

                //increment cnt when lo % 8 goes from 4->5, decrement cnt when lo % 8 goes from 5->4
                cnt_up <= 2'h0;
                if (aux & (lo==2'h1) & (lo_prev==2'h0)) cnt_up <= 2'h1;     //+1
                if (aux & (lo==2'h0) & (lo_prev==2'h1)) cnt_up <= 2'h3;     //-1

                //cnt_up is 1 clock cycle late compared to lo, so cnt is 2 clocks late
                cnt <= cnt + { {($bits(cnt)-1){cnt_up[1]}} , cnt_up[0] };   //sign extend before adding

                //since we are checking for lo % 8 == 1 to deassert valid_out and the decrement happens when lo % 8 == 4, then cnt_at_target can be up to 3 clocks late
                //upper bits of cnt_at_target can be even later since it would take lots of time from lo == 12 (cnt decrementing from 2) to lo == 1 (cnt_at_target must be asserted)
                if (WIDTH <= 4) begin   //cnt has sufficiently few bits that cnt_at_target is still only a single 6-lut
                    cnt_at_target <= (cnt=='h0) & ~(aux ^ flip_aux);        //cnt is 0 and next state of aux is 0
                    approx_full <= &cnt;                                    //cnt is max
                end
                else begin              //except for cnt[0], the upper bits are allowed to be 1 clock cycle late, use this to register some helpers
                    cnt_upper_all_zeros <= (cnt[$bits(cnt)-1:1]=='h0);      //could split into 6-bit sections
                    cnt_at_target <= cnt_upper_all_zeros & ~cnt[0] & ~(aux ^ flip_aux);  //up to 3 sections from cnt_upper_all_zeros can be merged here and still be only a single 6-lut
                    cnt_all_ones <= &cnt;                                   //could split into 6-bit sections
                    approx_full <= cnt_all_ones;                            //up to 6 sections of cnt_all_ones can be merged here and still only be a single 6-lut
                end

                //valid_out asserts if any write, deasserts if occupancy==1 and reading
                if (valid_in) valid_out <= 1'b1;
                else if (~stall_in & (lo==2'h1) & cnt_at_target) valid_out <= 1'b0;

                //in order to get an exact tracking of full, we basically do the same thing as valid_out but with reads and writes having an opposite effect
                //we start the inverse occupancy at DEPTH and reads decrement it down to 0, hence the different values based on DEPTH that lo is compared against
                flip_inv_aux <= ((lo==((DEPTH+2)&3)) & ~stall_in & valid_out & ~valid_in) | ((lo==((DEPTH+1)&3)) & ~(~stall_in & valid_out) & valid_in);
                inv_aux <= inv_aux ^ flip_inv_aux;

                inv_cnt_up <= 2'h0;
                if (inv_aux & (lo==((DEPTH)&3)) & (lo_prev==((DEPTH+3)&3))) inv_cnt_up <= 2'h3; //-1
                if (inv_aux & (lo==((DEPTH+3)&3)) & (lo_prev==((DEPTH)&3))) inv_cnt_up <= 2'h1; //+1

                inv_cnt <= inv_cnt + { {($bits(inv_cnt)-1){inv_cnt_up[1]}} , inv_cnt_up[0] };   //sign extend before adding

                if (WIDTH <= 4) begin
                    inv_cnt_at_target <= (inv_cnt=='h0) & ~(inv_aux ^ flip_inv_aux);
                end
                else begin
                    inv_cnt_upper_all_zeros <= (inv_cnt[$bits(inv_cnt)-1:1]=='h0);
                    inv_cnt_at_target <= inv_cnt_upper_all_zeros & ~inv_cnt[0] & ~(inv_aux ^ flip_inv_aux);
                end

                if (ALLOW_FULL_WRITE) begin
                    if (~stall_in & ~valid_in) exact_full <= 1'b0;
                    else if (valid_in & stall_in & (lo==((DEPTH+3)&3)) & inv_cnt_at_target) exact_full <= 1'b1;
                end
                else begin
                    if (~stall_in) exact_full <= 1'b0;
                    else if (valid_in & (lo==((DEPTH+3)&3)) & inv_cnt_at_target) exact_full <= 1'b1;
                end

                if (~sclrn) begin
                    lo <= 2'h0;
                    aux <= 1'b0;
                    inv_aux <= ((DEPTH+1)/4) & 1;
                    cnt <= 0;
                    inv_cnt <= (DEPTH+3)/8;
                    valid_out <= 1'b0;
                    approx_full <= 1'b0;
                    exact_full <= 1'b0;
                end
            end
        end
        assign full = (STRICT_DEPTH) ? exact_full : approx_full;
    end
    endgenerate

endmodule

`default_nettype wire
