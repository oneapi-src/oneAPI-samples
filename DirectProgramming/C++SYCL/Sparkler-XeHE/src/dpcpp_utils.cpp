/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#include "dpcpp_utils.h"

namespace xehe {
    namespace dpcpp {

        namespace sycl = cl::sycl;

        const uint32_t intel_vendor_id = 0x8086U;

        class intel_gpu_selector : public sycl::device_selector {
            virtual int operator()(const sycl::device &device) const override {
                if (device.is_gpu() && device.get_info<cl::sycl::info::device::vendor_id>() == intel_vendor_id) {
                    return 10000;
                }

                return -1;
            }
        };

        Context::Context(bool profiling_enabled /* = false */, bool select_gpu /* = true */) {
            if (igpu) {
#if 0
                auto exception_handler = [] (sycl::exception_list exceptions) {
            for (std::exception_ptr const& e : exceptions) {
                try {
                    std::rethrow_exception(e);
                } catch (cl::sycl::exception const& e) {
                    std::cout << "Caught asynchronous SYCL exception:" << std::endl;
                    std::cout << e.what() << std::endl;
                }
            }
        };
#endif
                _queues.clear();
                generate_queue(profiling_enabled, select_gpu);
                std::cout << "Running on \033[93m"
                          << _queues[0].get_device().get_info<sycl::info::device::name>()
                          << "\033[0m"
                          << std::endl;
            }

        }

        uint32_t Context::tiles() {
            return _queues.size();
        }

        const sycl::queue& Context::queue(uint32_t idx) const {
            return _queues[idx];
        }

        void Context::generate_queue(bool profiling_enabled, bool select_gpu){
            if (select_gpu) {
                // auto dev_selector = intel_gpu_selector();
                sycl::device RootDevice = sycl::device(intel_gpu_selector());
                std::cout << "RootDevice: " << RootDevice.get_info<sycl::info::device::name>() << std::endl;
                std::vector<sycl::device> SubDevices;
#ifndef WIN32
                try
                {
                    SubDevices = RootDevice.create_sub_devices<
                        sycl::info::partition_property::partition_by_affinity_domain>(
                            sycl::info::partition_affinity_domain::next_partitionable);
                }
                catch (...)
#endif
                {
                    std::cout << "Sub_devices are not supported\n";
                    
                    SubDevices.push_back(RootDevice);

                }

                sycl::context C(SubDevices);
                for (auto &D : SubDevices) {
                    sycl::queue q;
                    if (!profiling_enabled)
                        q = sycl::queue(C, D, sycl::property::queue::in_order());
                        //_queue = sycl::queue(dev_selector);
                    else
                        q = sycl::queue(C, D, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
                    _queues.push_back(q);
                }
            } else {
                sycl::queue q;
                auto dev_selector = sycl::cpu_selector();
                if (!profiling_enabled)
                    q = sycl::queue(dev_selector);
                else
                    q = sycl::queue(dev_selector, sycl::property::queue::enable_profiling());
                _queues.push_back(q);
            }
        }

        cl::sycl::queue create_queue(bool profiling_enabled, bool select_gpu){
            cl::sycl::queue q;
            if (select_gpu) {
                auto dev_selector = intel_gpu_selector();
                if (!profiling_enabled)
                    q = sycl::queue(dev_selector, sycl::property::queue::in_order());
                    //_queue = sycl::queue(dev_selector);
                else
                    q = sycl::queue(dev_selector, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
            } else {
                auto dev_selector = sycl::cpu_selector();
                if (!profiling_enabled)
                    q = sycl::queue(dev_selector);
                else
                    q = sycl::queue(dev_selector, sycl::property::queue::enable_profiling());
            }
            return q;
        }
    }
}