//==============================================================
// Copyright © 2024 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off

#include<sycl/sycl.hpp>
#include<iostream>

int main() {
// Snippet begin 1
    auto plat = sycl::platform(sycl::gpu_selector_v);
    auto devs = plat.get_devices();
    auto ctxt = sycl::context(devs);

    if (devs.size() < 2) {
       std::cerr << "No 2 GPU devices found" << std::endl;
       return -1;
    }

    std::cout << devs.size() << " GPU devices are found and 2 will be used" << std::endl;
    sycl::queue q[2];
    q[0] = sycl::queue(ctxt, devs[0], {sycl::property::queue::in_order()});
    q[1] = sycl::queue(ctxt, devs[1], {sycl::property::queue::in_order()});
// Snippet end 1

// Snippet begin 2
    constexpr size_t gsize = 1024 * 1024 * 1024L;
    float *ha = (float *)(malloc(gsize * sizeof(float)));
    float *hb = (float *)(malloc(gsize * sizeof(float)));
    float *hc = (float *)(malloc(gsize * sizeof(float)));

    for (size_t i = 0; i < gsize; i++) {
        ha[i] = float(i);
        hb[i] = float(i + gsize);
    }

    float *da[2];
    float *db[2];
    float *dc[2];

    size_t lsize = gsize / 2;

    da[0] = (float *)sycl::malloc_device<float>(lsize, q[0]);
    db[0] = (float *)sycl::malloc_device<float>(lsize, q[0]);
    dc[0] = (float *)sycl::malloc_device<float>(lsize, q[0]);
    q[0].memcpy(da[0], ha, lsize);
    q[0].memcpy(db[0], hb, lsize);

    da[1] = (float *)sycl::malloc_device<float>((lsize + gsize % 2), q[1]);
    db[1] = (float *)sycl::malloc_device<float>((lsize + gsize % 2), q[1]);
    dc[1] = (float *)sycl::malloc_device<float>((lsize + gsize % 2), q[1]);
    q[1].memcpy(da[1], ha + lsize, lsize + gsize % 2);
    q[1].memcpy(db[1], hb + lsize, lsize + gsize % 2);

    q[0].wait();
    q[1].wait();
// Snippet end 2

    std::cout << "Offloading work to 2 devices" << std::endl;

// Snippet begin 3
    for (int i = 0; i < 16; i ++) {
        q[0].parallel_for(sycl::nd_range<1>(lsize, 1024),[=](auto idx) {
            int ind = idx.get_global_id();
            dc[0][ind] = da[0][ind] + db[0][ind];
        });
        q[1].parallel_for(sycl::nd_range<1>(lsize + gsize % 2, 1024),[=](auto idx) {
            int ind = idx.get_global_id();
            dc[1][ind] = da[1][ind] + db[1][ind];
        });
    }

    q[0].wait();
    q[1].wait();

    std::cout << "Offloaded work completed" << std::endl;

    q[0].memcpy(hc, dc[0], lsize);
    q[1].memcpy(hc + lsize, dc[1], lsize + gsize % 2);

    q[0].wait();
    q[1].wait();
// Snippet end 3

    sycl::free(da[0], q[0]);
    sycl::free(db[0], q[0]);
    sycl::free(dc[0], q[0]);
    sycl::free(da[1], q[1]);
    sycl::free(db[1], q[1]);
    sycl::free(dc[1], q[1]);

    free(ha);
    free(hb);
    free(hc);

    return 0;
}
