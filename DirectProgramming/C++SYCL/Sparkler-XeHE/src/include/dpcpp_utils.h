/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef XeHE_DPCPP_UTILS_H
#define XeHE_DPCPP_UTILS_H

#ifdef __JETBRAINS_IDE__
// Stuff that only clion will see goes here
#define BUILD_WITH_IGPU
#endif

//#include <memory>
#include "lib_utils.h"
#if defined(BUILD_WITH_IGPU) || defined(SEAL_USE_INTEL_GPU)
namespace xehe {
    namespace dpcpp {
        class Context {
            bool igpu = true;
            std::vector<cl::sycl::queue> _queues;
            void generate_queue(bool profiling_enabled = false, bool select_gpu = true);

        public:
            Context(bool profiling_enabled = true, bool select_gpu = true);
            uint32_t tiles();
            const cl::sycl::queue& queue(uint32_t idx = 0) const;
        };

    }
}
#endif

#endif //XeHE_DPCPP_UTILS_H
