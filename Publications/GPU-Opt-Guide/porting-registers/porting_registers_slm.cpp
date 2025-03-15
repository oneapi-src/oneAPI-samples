// Snippet begin
#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>

#define NREGISTERS 80

#ifndef SIMD16
#define SIMD_SIZE 32
#else
#define SIMD_SIZE 16
#endif

class my_type {
public:
  my_type(double x0, double x1, double x2, double x3, double x4, double x5,
          double x6, double x7)
      : x0_(x0), x1_(x1), x2_(x2), x3_(x3), x4_(x4), x5_(x5), x6_(x6), x7_(x7) {
  }

  my_type(double const *const vals)
      : x0_(vals[0]), x1_(vals[1]), x2_(vals[2]), x3_(vals[3]), x4_(vals[4]),
        x5_(vals[5]), x6_(vals[6]), x7_(vals[7]) {}

  my_type operator+(const my_type &rhs) {
    return my_type(rhs.x0_ + x0_, rhs.x1_ + x1_, rhs.x2_ + x2_, rhs.x3_ + x3_,
                   rhs.x4_ + x4_, rhs.x5_ + x5_, rhs.x6_ + x6_, rhs.x7_ + x7_);
  }

  void WriteBack(double *const vals) {
    vals[0] += x0_;
    vals[1] += x1_;
    vals[2] += x2_;
    vals[3] += x3_;
    vals[4] += x4_;
    vals[5] += x5_;
    vals[6] += x6_;
    vals[7] += x7_;
  }

  void Load(double const *const vals) {
    x0_ = vals[0];
    x1_ = vals[1];
    x2_ = vals[2];
    x3_ = vals[3];
    x4_ = vals[4];
    x5_ = vals[5];
    x6_ = vals[6];
    x7_ = vals[7];
  }

private:
  double x0_, x1_, x2_, x3_, x4_, x5_, x6_, x7_;
};

int main(int argc, char **) {
  sycl::queue Q(sycl::gpu_selector_v);
  size_t nsubgroups;
  if (argc > 1)
    nsubgroups = 10;
  else
    nsubgroups = 8;

  const size_t ARR_SIZE = nsubgroups * SIMD_SIZE * NREGISTERS;
  double *val_in = sycl::malloc_shared<double>(ARR_SIZE, Q);
  double *val_out = sycl::malloc_shared<double>(ARR_SIZE, Q);

  std::cout << "Using simd size " << SIMD_SIZE << std::endl;

  Q.parallel_for(ARR_SIZE, [=](auto it) {
     val_in[it.get_id()] = argc * it.get_id();
   }).wait();

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int rep = 0; rep < 1000; rep++) {
    Q.submit([&](auto &h) {
       sycl::local_accessor<double, 1> slm(sycl::range(32 * 8), h);

       h.parallel_for(sycl::nd_range<1>(nsubgroups * SIMD_SIZE, 32),
                      [=](auto it) [[intel::reqd_sub_group_size(SIMD_SIZE)]] {
                        const int id = it.get_global_linear_id();
                        const int local_id = it.get_local_id(0);
                        for (int i = 0; i < 8; i++) {
                          slm[i + local_id * 8] = 0.0;
                        }
#ifdef SIMD16
                        it.barrier(sycl::access::fence_space::local_space);
#endif

                        double const *const val_offs = val_in + id * NREGISTERS;

                        my_type v0(val_offs + 0);
                        my_type v1(val_offs + 8);
                        my_type v2(val_offs + 16);
                        my_type v3(val_offs + 24);
                        my_type v4(val_offs + 32);
                        my_type v5(val_offs + 40);
                        my_type v6(val_offs + 48);
                        my_type v7(val_offs + 56);
                        my_type v8(val_offs + 64);
                        my_type v9(val_offs + 72);

                        my_type p0 = v0 + v5;
                        v0.WriteBack(&slm[local_id * 8]);
#ifdef SIMD16
                        it.barrier(sycl::access::fence_space::local_space);
#endif
                        v0.Load(&slm[32 * 8 - 8 - local_id * 8]);
                        my_type p1 = v1 + v6;
                        my_type p2 = v2 + v7;
                        my_type p3 = v3 + v8;
                        my_type p4 = v4 + v9;

                        p0 = p0 + p1 + v5;
                        p2 = p2 + p3 + v6;
                        p4 = p4 + p1 + v7;
                        p3 = p3 + p2 + v8;
                        p2 = p0 + p4 + v9;

                        p1 = p1 + v0;

                        double *const vals_out_offs = val_out + id * NREGISTERS;
                        p0.WriteBack(vals_out_offs);
                        p1.WriteBack(vals_out_offs + 8);
                        p2.WriteBack(vals_out_offs + 16);
                        p3.WriteBack(vals_out_offs + 24);
                        p4.WriteBack(vals_out_offs + 32);
                        v0.WriteBack(vals_out_offs + 40);
                        v1.WriteBack(vals_out_offs + 48);
                        v2.WriteBack(vals_out_offs + 56);
                        v3.WriteBack(vals_out_offs + 64);
                        v4.WriteBack(vals_out_offs + 72);
                      });
     }).wait();
  }
  auto end_time = std::chrono::high_resolution_clock::now();

  std::cout << "Took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     start_time)
                   .count()
            << " microseconds" << std::endl;

  sycl::free(val_in, Q);
  sycl::free(val_out, Q);

  return 0;
}
// Snippet end
