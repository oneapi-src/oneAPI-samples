/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XeHE_INLINE_KERNELS_H
#define XeHE_INLINE_KERNELS_H

//x is var size
//y is simd size

#define U_VAR_TYPE(x) ((x == 4)? "ud": "uq")

#define VAR_TYPE(x) ((x == 4)? "d": "q")

#define VAR_align(x) ((x == 4)? "hword": "qword")

#define STR_HELPER(y) #y
#define STR(y) STR_HELPER(y)

// #define ADDMOD_STR(x,y) "\n" \
//                             ".decl temp0 v_type=G type=" U_VAR_TYPE(x)" num_elts= " STR(y)" align=" VAR_align(x)" alias=<%0, 0>\n" \
//                             ".decl temp1 v_type=G type=" U_VAR_TYPE(x) "num_elts=1 align=hword alias=<%3, 0>\n" \
//                             ".decl temp2 v_type=G type=" VAR_TYPE(x)" num_elts= " STR(y)" align=" VAR_align(x)"\n" \
//                             ".decl P1 v_type=P num_elts=" STR(x)"\n" \
//                             "\n" \
//                             "add (M1, " STR(y)") %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0>\n" \
//                             "cmp.lt (M1, " STR(y)") P1 temp0(0,0)<1;1,0> temp1(0,0)<0;1,0>\n" \
//                             "(P1) sel (M1, " STR(y)") temp2(0,0)<1> 0x0:d %3(0,0)<0;1,0>\n"\
//                             "add (M1, " STR(y)") %0(0,0)<1> %0(0,0)<1;1,0> (-)temp2(0,0)<1;1,0>\n" \
//                             "\n"

#define ADD_STR(y)  "add (M1, " STR(y) ") %0(0, 0)<1> %1(0, 0)<1;1,0> %2(0, 0)<1;1,0>\n"


#define ADDMOD_32_STR(y)    "{\n" \
                            ".decl temp0 v_type=G type=ud num_elts=" STR(y) " align=hword alias=<%0, 0>\n" \
                            ".decl temp1 v_type=G type=ud num_elts=1 align=hword alias=<%3, 0>\n" \
                            ".decl temp2 v_type=G type=d num_elts=" STR(y) " align=hword\n" \
                            ".decl P1 v_type=P num_elts=" STR(y) "\n" \
                            "\n" \
                            "add (M1, " STR(y) ") %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0>\n" \
                            "cmp.lt (M1, " STR(y) ") P1 temp0(0,0)<1;1,0> temp1(0,0)<0;1,0>\n" \
                            "(P1) sel (M1, " STR(y) ") temp2(0,0)<1> 0x0:d %3(0,0)<0;1,0>\n"\
                            "add (M1, " STR(y) ") %0(0,0)<1> %0(0,0)<1;1,0> (-)temp2(0,0)<1;1,0>\n" \
                            "}\n"

#define ADDMOD_64_STR(y)    "{\n" \
                            ".decl temp0 v_type=G type=uq num_elts=" STR(y) " align=qword alias=<%0, 0>\n" \
                            ".decl temp1 v_type=G type=uq num_elts=1 align=qword alias=<%3, 0>\n" \
                            ".decl temp2 v_type=G type=q num_elts=" STR(y) " align=qword\n" \
                            ".decl P1 v_type=P num_elts=" STR(y) "\n"\
                            "\n" \
                            "add (M1, " STR(y) ") %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0>\n" \
                            "cmp.lt (M1, " STR(y) ") P1 temp0(0,0)<1;1,0> temp1(0,0)<0;1,0>\n" \
                            "(P1) sel (M1, " STR(y) ") temp2(0,0)<1> 0x0:d %3(0,0)<0;1,0>\n"\
                            "add (M1, " STR(y) ") %0(0,0)<1> %0(0,0)<1;1,0> (-)temp2(0,0)<1;1,0>\n" \
                            "}\n"

#define ADDMOD_OPT_32_STR(y)   "{\n" \
                                ".decl temp0%= v_type=G type=ud num_elts=" STR(y) " align=hword alias=<%0, 0>\n"   \
                                ".decl temp1%= v_type=G type=ud num_elts=" STR(y) " align=hword alias=<%3, 0>\n"   \
                                ".decl temp2%= v_type=P num_elts=" STR(y) "\n"    \
                                "\n"    \
                                "add (M1, " STR(y) ") %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0>\n"    \
                                "cmp.ge (M1, " STR(y) ") temp2%= temp0%=(0,0)<1;1,0> temp1%=(0,0)<1;1,0>\n"   \
                                "(temp2%=) add (M1, " STR(y) ") %0(0,0)<1> %0(0,0)<1;1,0> (-)%3(0,0)<1;1,0>\n"    \
                                "}\n"

#define ADDMOD_OPT_64_STR(y)    "{\n" \
                                ".decl temp0%= v_type=G type=uq num_elts=" STR(y) " align=qword alias=<%0, 0>\n"   \
                                ".decl temp1%= v_type=G type=uq num_elts=" STR(y) " align=qword alias=<%3, 0>\n"   \
                                ".decl temp2%= v_type=P num_elts=" STR(y) "\n"    \
                                "\n"    \
                                "add (M1, " STR(y) ") %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0>\n"    \
                                "cmp.ge (M1, " STR(y) ") temp2%= temp0%=(0,0)<1;1,0> temp1%=(0,0)<1;1,0>\n"   \
                                "(temp2%=) add (M1, " STR(y) ") %0(0,0)<1> %0(0,0)<1;1,0> (-)%3(0,0)<1;1,0>\n"    \
                                "}\n"

#define MUL_UINT_OPT_64_STR(y)  "{\n" \
                                ".decl temp1%= v_type=G type=ud num_elts=" STR(y) " align=hword\n" \
                                ".decl temp2%= v_type=G type=ud num_elts=" STR(y) " align=hword\n" \
                                "mov (M1, " STR(y) ") temp1%=(0,0)<1> %1(0,0)<1;1,0>\n" \
                                "mov (M1, " STR(y) ") temp2%=(0,0)<1> %2(0,0)<1;1,0>\n" \
                                "mul (M1, " STR(y) ") %0(0,0)<1> temp1%=(0,0)<1;1,0> temp2%=(0,0)<1;1,0>\n" \
                                "}\n"

#endif  /*XeHE_INLINE_KERNELS_H*/