//==============================================================
// Copyright © 2024 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off

#include<sycl/sycl.hpp>
#include<iostream>

int main() {
    auto plat = sycl::platform(sycl::gpu_selector_v);
    auto devs = plat.get_devices();
    auto ctxt = sycl::context(devs);

    if (devs.size() < 1) {
       std::cerr << "No GPU devices found" << std::endl;
       return -1;
    }

    std::cout << devs.size() << " GPU devices are found and 2 will be used" << std::endl;
    sycl::queue q(ctxt, devs[0], {sycl::property::queue::in_order()});

    constexpr size_t gsize = 1024 * 1024 * 1024L;
    float *ha = (float *)(malloc(gsize * sizeof(float)));
    float *hb = (float *)(malloc(gsize * sizeof(float)));
    float *hc = (float *)(malloc(gsize * sizeof(float)));

    for (size_t i = 0; i < gsize; i++) {
        ha[i] = float(i);
        hb[i] = float(i + gsize);
    }

// Snippet begin
    float *da;
    float *db;
    float *dc;

    da = (float *)sycl::malloc_device<float>(gsize, q);
    db = (float *)sycl::malloc_device<float>(gsize, q);
    dc = (float *)sycl::malloc_device<float>(gsize, q);
    q.memcpy(da, ha, gsize);
    q.memcpy(db, hb, gsize);

    q.wait();

    std::cout << "Offloading work to 1 device" << std::endl;

    for (int i = 0; i < 16; i ++) {
        q.parallel_for(sycl::nd_range<1>(gsize, 1024),[=](auto idx) {
            int ind = idx.get_global_id();
            dc[ind] = da[ind] + db[ind];
        });
    }

    q.wait();

    std::cout << "Offloaded work completed" << std::endl;

    q.memcpy(hc, dc, gsize);
// Snippet end

    sycl::free(da, q);
    sycl::free(db, q);
    sycl::free(dc, q);

    free(ha);
    free(hb);
    free(hc);

    return 0;
}
