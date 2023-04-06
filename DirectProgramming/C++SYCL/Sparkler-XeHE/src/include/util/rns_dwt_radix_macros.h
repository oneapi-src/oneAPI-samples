/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef _RNS_DWT_RADIX_MACROS_H_
#define _RNS_DWT_RADIX_MACROS_H_

#define ld_global_mem_to_reg_1(offset) \
  dx[offset] = *(x[offset]);
#define ld_global_mem_to_reg_2(offset) \
  ld_global_mem_to_reg_1(offset) \
  ld_global_mem_to_reg_1(offset+1)
#define ld_global_mem_to_reg_4(offset) \
  ld_global_mem_to_reg_2(offset) \
  ld_global_mem_to_reg_2(offset+2)
#define ld_global_mem_to_reg_8(offset) \
  ld_global_mem_to_reg_4(offset) \
  ld_global_mem_to_reg_4(offset+4)
#define ld_global_mem_to_reg_16(offset) \
  ld_global_mem_to_reg_8(offset) \
  ld_global_mem_to_reg_8(offset+8)
#define ld_local_mem_to_reg_1(offset) \
  dx[offset] = *(x_local[offset]);
#define ld_local_mem_to_reg_2(offset) \
  ld_local_mem_to_reg_1(offset) \
  ld_local_mem_to_reg_1(offset+1)
#define ld_local_mem_to_reg_4(offset) \
  ld_local_mem_to_reg_2(offset) \
  ld_local_mem_to_reg_2(offset+2)
#define ld_local_mem_to_reg_8(offset) \
  ld_local_mem_to_reg_4(offset) \
  ld_local_mem_to_reg_4(offset+4)
#define ld_local_mem_to_reg_16(offset) \
  ld_local_mem_to_reg_8(offset) \
  ld_local_mem_to_reg_8(offset+8)
#define st_reg_to_global_mem_1(offset) \
  *(x[offset]) = dx[offset];
#define st_reg_to_global_mem_2(offset) \
  st_reg_to_global_mem_1(offset) \
  st_reg_to_global_mem_1(offset+1)
#define st_reg_to_global_mem_4(offset) \
  st_reg_to_global_mem_2(offset) \
  st_reg_to_global_mem_2(offset+2)
#define st_reg_to_global_mem_8(offset) \
  st_reg_to_global_mem_4(offset) \
  st_reg_to_global_mem_4(offset+4)
#define st_reg_to_global_mem_16(offset) \
  st_reg_to_global_mem_8(offset) \
  st_reg_to_global_mem_8(offset+8)
#define local_last_round_processing_1(offset) \
  dx[offset] = *(x_local[offset]);\
  dx[offset] -= (dx[offset] >= two_times_modulus) ? two_times_modulus : 0;\
  dx[offset] = dx[offset] - ((dx[offset] >= modulus) ? modulus : 0);
#define local_last_round_processing_2(offset) \
  local_last_round_processing_1(offset) \
  local_last_round_processing_1(offset+1)
#define local_last_round_processing_4(offset) \
  local_last_round_processing_2(offset) \
  local_last_round_processing_2(offset+2)
#define local_last_round_processing_8(offset) \
  local_last_round_processing_4(offset) \
  local_last_round_processing_4(offset+4)
#define local_last_round_processing_16(offset) \
  local_last_round_processing_8(offset) \
  local_last_round_processing_8(offset+8)
#define ld_roots_2(ptr_src, ptr_dst, rounds) \
  ptr_dst[0] = ptr_src[(i1) + (rounds) + (global_offset)];\
  ptr_dst[1] = ptr_src[(i2) + (rounds) + (global_offset)];
#define ld_roots_4(ptr_src, ptr_dst, rounds) \
  ld_roots_2(ptr_src, ptr_dst, rounds)\
  ptr_dst[2] = ptr_src[(i3) + (rounds) + (global_offset)];\
  ptr_dst[3] = ptr_src[(i4) + (rounds) + (global_offset)];
#define ld_roots_8(ptr_src, ptr_dst, rounds) \
  ld_roots_4(ptr_src, ptr_dst, rounds)\
  ptr_dst[4] = ptr_src[(i5) + (rounds) + (global_offset)];\
  ptr_dst[5] = ptr_src[(i6) + (rounds) + (global_offset)];\
  ptr_dst[6] = ptr_src[(i7) + (rounds) + (global_offset)];\
  ptr_dst[7] = ptr_src[(i8) + (rounds) + (global_offset)];
#define ld_roots_relin_2(ptr_src, ptr_dst, rounds) \
  ptr_dst[0] = ptr_src[(i1) + (rounds) + (kk_mul_n)];\
  ptr_dst[1] = ptr_src[(i2) + (rounds) + (kk_mul_n)];
#define ld_roots_relin_4(ptr_src, ptr_dst, rounds) \
  ld_roots_relin_2(ptr_src, ptr_dst, rounds)\
  ptr_dst[2] = ptr_src[(i3) + (rounds) + (kk_mul_n)];\
  ptr_dst[3] = ptr_src[(i4) + (rounds) + (kk_mul_n)];
#define ld_roots_relin_8(ptr_src, ptr_dst, rounds) \
  ld_roots_relin_4(ptr_src, ptr_dst, rounds)\
  ptr_dst[4] = ptr_src[(i5) + (rounds) + (kk_mul_n)];\
  ptr_dst[5] = ptr_src[(i6) + (rounds) + (kk_mul_n)];\
  ptr_dst[6] = ptr_src[(i7) + (rounds) + (kk_mul_n)];\
  ptr_dst[7] = ptr_src[(i8) + (rounds) + (kk_mul_n)];
#define init_global_ptr_base(gap_) \
  x[0] = values_ + global_offset + offset + poly_offset;\
  x[1] = x[0] + gap_;
#define init_global_ptr_4(gap_) \
  init_global_ptr_base(gap_)\
  x[2] = x[1] + gap_;\
  x[3] = x[2] + gap_;
#define init_global_ptr_8(gap_) \
  init_global_ptr_4(gap_)\
  x[4] = x[3] + gap_;\
  x[5] = x[4] + gap_;\
  x[6] = x[5] + gap_;\
  x[7] = x[6] + gap_;
#define init_global_ptr_16(gap_) \
  init_global_ptr_8(gap_)\
  x[8] = x[7] + gap_;\
  x[9] = x[8] + gap_;\
  x[10] = x[9] + gap_;\
  x[11] = x[10] + gap_;\
  x[12] = x[11] + gap_;\
  x[13] = x[12] + gap_;\
  x[14] = x[13] + gap_;\
  x[15] = x[14] + gap_;
#define init_global_ptr_inv_relin_base(gap_) \
  x[0] = values_ + offset + poly_offset;\
  x[1] = x[0] + gap_;
#define init_global_ptr_inv_relin_4(gap_) \
  init_global_ptr_inv_relin_base(gap_)\
  x[2] = x[1] + gap_;\
  x[3] = x[2] + gap_;
#define init_global_ptr_inv_relin_8(gap_) \
  init_global_ptr_inv_relin_4(gap_)\
  x[4] = x[3] + gap_;\
  x[5] = x[4] + gap_;\
  x[6] = x[5] + gap_;\
  x[7] = x[6] + gap_;
#define init_global_ptr_inv_relin_16(gap_) \
  init_global_ptr_inv_relin_8(gap_)\
  x[8] = x[7] + gap_;\
  x[9] = x[8] + gap_;\
  x[10] = x[9] + gap_;\
  x[11] = x[10] + gap_;\
  x[12] = x[11] + gap_;\
  x[13] = x[12] + gap_;\
  x[14] = x[13] + gap_;\
  x[15] = x[14] + gap_;
#define init_local_ptr_base(gap_) \
  x_local[0] = ptr_.get_pointer() + offset_local;\
  x_local[1] = x_local[0] + gap_;
#define init_local_ptr_4(gap_) \
  init_local_ptr_base(gap_)\
  x_local[2] = x_local[1] + gap_;\
  x_local[3] = x_local[2] + gap_;
#define init_local_ptr_8(gap_) \
  init_local_ptr_4(gap_)\
  x_local[4] = x_local[3] + gap_;\
  x_local[5] = x_local[4] + gap_;\
  x_local[6] = x_local[5] + gap_;\
  x_local[7] = x_local[6] + gap_;
#define init_local_ptr_16(gap_) \
  init_local_ptr_8(gap_)\
  x_local[8] = x_local[7] + gap_;\
  x_local[9] = x_local[8] + gap_;\
  x_local[10] = x_local[9] + gap_;\
  x_local[11] = x_local[10] + gap_;\
  x_local[12] = x_local[11] + gap_;\
  x_local[13] = x_local[12] + gap_;\
  x_local[14] = x_local[13] + gap_;\
  x_local[15] = x_local[14] + gap_;
#define butterfly_ntt_reg2reg(offset_x, offset_y, offset_root) \
  u = dwt_guard(dx[offset_x], two_times_modulus);\
  v = dwt_mul_root(dx[offset_y], r_op[offset_root], r_quo[offset_root], modulus);\
  dx[offset_x] = dwt_add(u, v);\
  dx[offset_y] = dwt_sub(u, v, two_times_modulus);
#define butterfly_ntt_reg2gmem(offset_x, offset_y, offset_root) \
  u = dwt_guard(dx[offset_x], two_times_modulus);\
  v = dwt_mul_root(dx[offset_y], r_op[offset_root], r_quo[offset_root], modulus);\
  *x[offset_x] = dwt_add(u, v);\
  *x[offset_y] = dwt_sub(u, v, two_times_modulus);
#define butterfly_ntt_reg2lmem(offset_x, offset_y, offset_root) \
  u = dwt_guard(dx[offset_x], two_times_modulus);\
  v = dwt_mul_root(dx[offset_y], r_op[offset_root], r_quo[offset_root], modulus);\
  *x_local[offset_x] = dwt_add(u, v);\
  *x_local[offset_y] = dwt_sub(u, v, two_times_modulus);
#define butterfly_inv_ntt_reg2reg(offset_x, offset_y, offset_root) \
  u = dx[offset_x];\
  v = dx[offset_y];\
  dx[offset_x] = dwt_guard(dwt_add(u, v), two_times_modulus);\
  dx[offset_y] = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op[offset_root], r_quo[offset_root], modulus);
#define butterfly_inv_ntt_reg2lmem(offset_x, offset_y, offset_root) \
  u = dx[offset_x];\
  v = dx[offset_y];\
  *x_local[offset_x] = dwt_guard(dwt_add(u, v), two_times_modulus);\
  *x_local[offset_y] = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op[offset_root], r_quo[offset_root], modulus);
#define butterfly_inv_ntt_reg2gmem(offset_x, offset_y, offset_root) \
  u = dx[offset_x];\
  v = dx[offset_y];\
  *x[offset_x] = dwt_guard(dwt_add(u, v), two_times_modulus);\
  *x[offset_y] = dwt_mul_root(dwt_sub(u, v, two_times_modulus), r_op[offset_root], r_quo[offset_root], modulus);
#define ld_local2reg_1(offset) \
  dx[offset] = *(x_local[offset]);
#define ld_local2reg_2(offset) \
  ld_local2reg_1(offset) \
  ld_local2reg_1(offset+1)
#define ld_local2reg_4(offset) \
  ld_local2reg_2(offset) \
  ld_local2reg_2(offset+2)
#define ld_local2reg_8(offset) \
  ld_local2reg_4(offset) \
  ld_local2reg_4(offset+4)
#define ld_local2reg_16(offset) \
  ld_local2reg_8(offset) \
  ld_local2reg_8(offset+8)
#define ld_local2reg_scale_1(offset) \
  dx[offset] = dwt_mul_scalar(dwt_guard(*(x_local[offset]), two_times_modulus), scalar_op, scalar_quo, modulus);\
  dx[offset] -= (dx[offset] >= modulus) ? modulus : 0;
#define ld_local2reg_scale_2(offset) \
  ld_local2reg_scale_1(offset) \
  ld_local2reg_scale_1(offset+1)
#define ld_local2reg_scale_4(offset) \
  ld_local2reg_scale_2(offset) \
  ld_local2reg_scale_2(offset+2)
#define ld_local2reg_scale_8(offset) \
  ld_local2reg_scale_4(offset) \
  ld_local2reg_scale_4(offset+4)
#define ld_local2reg_scale_16(offset) \
  ld_local2reg_scale_8(offset) \
  ld_local2reg_scale_8(offset+8)
#define st_reg2gmem_scale_1(offset) \
  dx[offset] = dwt_mul_scalar(dwt_guard(dx[offset], two_times_modulus), scalar_op, scalar_quo, modulus);\
  dx[offset] -= (dx[offset] >= modulus) ? modulus : 0;\
  *x[offset] = dx[offset];
#define st_reg2gmem_scale_2(offset) \
  st_reg2gmem_scale_1(offset) \
  st_reg2gmem_scale_1(offset+1)
#define st_reg2gmem_scale_4(offset) \
  st_reg2gmem_scale_2(offset) \
  st_reg2gmem_scale_2(offset+2)
#define st_reg2gmem_scale_8(offset) \
  st_reg2gmem_scale_4(offset) \
  st_reg2gmem_scale_4(offset+4)
#define st_reg2gmem_scale_16(offset) \
  st_reg2gmem_scale_8(offset) \
  st_reg2gmem_scale_8(offset+8)
#define compute_ind_2(log_gap) \
  i1 = (ind1) >> ((log_gap));\
  i2 = (ind2) >> ((log_gap));
#define compute_ind_4(log_gap) \
  compute_ind_2(log_gap) \
  i3 = (ind3) >> ((log_gap));\
  i4 = (ind4) >> ((log_gap));
#define compute_ind_8(log_gap) \
  compute_ind_4(log_gap) \
  i5 = (ind5) >> ((log_gap));\
  i6 = (ind6) >> ((log_gap));\
  i7 = (ind7) >> ((log_gap));\
  i8 = (ind8) >> ((log_gap));
#define compute_ind_local_2(log_gap) \
  i1_local = (ind1_local) >> ((log_gap));\
  i2_local = (ind2_local) >> ((log_gap));
#define compute_ind_local_4(log_gap) \
  compute_ind_local_2(log_gap) \
  i3_local = (ind3_local) >> ((log_gap));\
  i4_local = (ind4_local) >> ((log_gap));
#define compute_ind_local_8(log_gap) \
  compute_ind_local_4(log_gap) \
  i5_local = (ind5_local) >> ((log_gap));\
  i6_local = (ind6_local) >> ((log_gap));\
  i7_local = (ind7_local) >> ((log_gap));\
  i8_local = (ind8_local) >> ((log_gap));
#endif//_RNS_DWT_RADIX_MACROS_H_
