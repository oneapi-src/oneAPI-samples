`timescale 1 ps / 1 ps
 
module byteswap_uint (
  input   clock,
  input   resetn,
  input   ivalid, 
  input   iready,
  output  ovalid,
  output  oready,
  input   [31:0]  datain,
  output  [31:0]  dataout);
 
  assign  ovalid = 1'b1;
  assign  oready = 1'b1;
  // clk, ivalid, iready, resetn are ignored
  assign dataout = {datain[15:0], datain[31:16]};
 
endmodule
