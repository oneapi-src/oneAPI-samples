module dsp_27x27u #(
    // To be replaced with the appropriate family
    // In this build system, cmake replaces the family automatically
    parameter FAMILY = "@FAMILY@",
    parameter LATENCY = 2,
    parameter AX_WIDTH = 27,
    parameter AY_WIDTH = 27,
    parameter RESULT_A_WIDTH = 54
) (
    input   clock,
    input   resetn,
    input   ivalid,
    input   iready,
    output  ovalid,
    output  oready,
    input [AX_WIDTH-1:0] ax,
    input [AY_WIDTH-1:0] ay,
    output [RESULT_A_WIDTH-1:0] resulta
);
 
initial begin
    if ((FAMILY == "Agilex") && ((LATENCY < 2) || (LATENCY > 4)))
        $fatal("(LATENCY < %d) || (LATENCY > %d): %d", 2, 4, LATENCY);
    if ((FAMILY == "Stratix 10") && ((LATENCY < 2) || (LATENCY > 4)))
        $fatal("(LATENCY < %d) || (LATENCY > %d): %d", 2, 4, LATENCY);
    if ((FAMILY == "Arria 10") && ((LATENCY < 2) || (LATENCY > 3)))
        $fatal("(LATENCY < %d) || (LATENCY > %d): %d", 2, 3, LATENCY);
end
 
generate
 
if (FAMILY == "Agilex") begin
    (* altera_attribute = "-name DSP_REGISTER_PACKING Disable" *)
    tennm_mac  #(
        .ax_width (AX_WIDTH),
        .ay_scan_in_width (AY_WIDTH),
        .operation_mode ("m27x27"),
        .ax_clken ("0"),
        .ay_scan_in_clken ("0"),
        .input_pipeline_clken ((LATENCY == 4) ? "0" : "no_reg"),
        .second_pipeline_clken ((LATENCY >= 3) ? "0" : "no_reg"),
        .output_clken ("0"),
        .scan_out_width (AY_WIDTH),
        .result_a_width (RESULT_A_WIDTH)
    ) dsp (
        .clr (2'b0),
        .ax (ax),
        .ay (ay),
        .clk (clock),
        .ena (3'b111),
        .resulta (resulta)
    );
end else if (FAMILY == "Stratix 10") begin
    (* altera_attribute = "-name DSP_REGISTER_PACKING Disable" *)
    fourteennm_mac  #(
        .ax_width (AX_WIDTH),
        .ay_scan_in_width (AY_WIDTH),
        .operation_mode ("m27x27"),
        .ax_clock ("0"),
        .ay_scan_in_clock ("0"),
        .input_pipeline_clock ((LATENCY >= 3) ? "0" : "none"),
        .second_pipeline_clock ((LATENCY == 4) ? "0" : "none"),
        .output_clock ("0"),
        .scan_out_width (AY_WIDTH),
        .result_a_width (RESULT_A_WIDTH)
    ) dsp (
        .clr (2'b0),
        .ax (ax),
        .ay (ay),
        .clk ({clock, clock, clock}),
        .ena (3'b111),
        .resulta (resulta)
    );
end else if (FAMILY == "Arria 10") begin
    (* altera_attribute = "-name DSP_REGISTER_PACKING Disable" *)
    twentynm_mac  #(
        .ax_width (AX_WIDTH),
        .ay_scan_in_width (AY_WIDTH),
        .operation_mode ("m27x27"),
        .ax_clock ("0"),
        .ay_scan_in_clock ("0"),
        .input_pipeline_clock ((LATENCY == 3) ? "0" : "none"),
        .output_clock ("0"),
        .scan_out_width (AY_WIDTH),
        .result_a_width (RESULT_A_WIDTH)
    ) dsp (
        .aclr (2'b0),
        .ax (ax),
        .ay (ay),
        .clk ({clock, clock, clock}),
        .ena (3'b111),
        .resulta (resulta)
    );
end else begin
    reg [RESULT_A_WIDTH-1:0] resulta_r[0:LATENCY-1];
    integer i;
    always @(posedge clock) begin
        resulta_r[0] <= ax * ay;
        for (i = 1; i < LATENCY; i=i+1) begin
            resulta_r[i] <= resulta_r[i-1];
        end
    end
    assign resulta = resulta_r[LATENCY-1];
end
    //avalon streaming interface
    assign  ovalid = 1'b1;
    assign  oready = 1'b1;
    // ivalid, iready are ignored 
endgenerate

endmodule
