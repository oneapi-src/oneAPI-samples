/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef XeHE_LIB_UTILS_H
#define XeHE_LIB_UTILS_H
#include <type_traits>
#include <cstdint>
#include <map>
#include <algorithm>
#include <fstream>
#include <iostream>

// XeHE basic definitions and parameters
#include "util/defines.h"


#if defined(BUILD_WITH_IGPU) || defined(SEAL_USE_INTEL_GPU)
#include <CL/sycl.hpp>
#endif //#if defined(BUILD_WITH_IGPU) || defined(SEAL_USE_INTEL_GPU)

namespace xehe {
    namespace dpcpp {
        template <typename T>
        T addm(T op1, T op2, T mod) {
            auto r = op1 + op2;
            auto signed_r = std::make_signed<T>(r);
            r = signed_r < 0 ? r + mod : r;
            return r;
        }

    }

#if 1
    typedef union
    {
        struct {
            uint16_t values16b[4];
        };
        struct {
            uint32_t values32b[2];
        };
        struct {
            uint32_t part[2];
        };
        struct {
            uint32_t low;
            uint32_t high;
        };
        uint64_t value64b;
    } w64_t;



    typedef union
    {

        struct {
            w64_t part[2];
        };
        struct {
            w64_t low;
            w64_t high;
        };
    } w128_t;
#else
    typedef struct {
        uint32_t part[2];
    } w64_t;

    typedef         struct {
        w64_t part[2];
    } w128_t;
#endif

#if defined(BUILD_WITH_IGPU) || defined(SEAL_USE_INTEL_GPU)

#include <CL/sycl.hpp>

#define _NO_WAIT_ false
#define _INF_COMP_ false
#define _INF_MEMORY_ false

    struct EventStats{
        std::string event_name;
        double avg_time;
        double percentage_of_op;
        int exec_num;
    };

    class EventCollector{
    public:
        // clear the events in the private member events_ for starting event collections from empty map.
        static void clear_events(){
            if (!activated_) return;
            events_.clear();
        }

        // add event with the kernel name to record profilling info; this call is internal only for adding events for kernel calls.
        static void add_event(std::string name, cl::sycl::event e){
            if (!activated_) return;
            if (events_.find(name)==events_.end()){
                events_[name]={e};
            }
            else{
                events_[name].push_back(e);
            }
        }

        // process the profilling info of all the events saved; print out the execution frequency and timing statistics.
        static void process_events(){
            if (!activated_) return;
            // TODO: POSSIBLE QUEUE.WAIT
            for (auto const& item : events_){
                std::string event_name = item.first;
                double avg_exec_time = .0;
                for (auto const& e : item.second){
                    auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
                    auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
                    avg_exec_time += end - start;
                }
                avg_exec_time /= (double) item.second.size()*1000.0;
                std::cout << "Kernel " << event_name << " executed " << item.second.size() << " times. Average execution time: " << avg_exec_time << "us" << std::endl;
            }
        }

        // add header to the export table with user specifying the parameters
        static void add_header(int poly_order, int data_bound, int delta_bits, int rns_size){
            export_table_ += "poly order: " + std::to_string(poly_order) + 
                             ", data bound: " + std::to_string(data_bound) + 
                             ", scale(log): " + std::to_string(delta_bits) + 
                             ", RNS base: " + std::to_string(rns_size) + "\n";
            export_table_ += "Operation\tAvg Operation Time(us)\tKernel Name\tAvg Kernel Time(us)\tKernel Executions\tPercentage of Internal Kernel(s)\tPercentage of Externel Operation\n";
        }

        /*** 
        process the profilling info of all the events saved; 
        combine them into one operation, statistics related to this operation is calculated;
        append the statistics to export_table_, events are sorted decending by their percentage of time.
        ***/
        static void add_operation(std::string op_name, double avg_external_time, int loops){
            if (!activated_) return;
            std::vector<EventStats> event_stats;
            export_table_ += op_name + "\t" + std::to_string(avg_external_time) + "\t";

            double total_kernel_time = 0.0;
            double total_percentable_of_op = 0.0;
            double total_NTT_time = 0.0;
            double total_percentable_of_NTT = 0.0;

            for (auto const& item : events_){
                EventStats cur_event;
                cur_event.event_name = item.first;
                double avg_exec_time = .0;
                for (auto const& e : item.second){
                    auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
                    auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
                    avg_exec_time += end - start;
                }
                avg_exec_time /= (double) item.second.size()*1000.0;
                cur_event.avg_time = avg_exec_time;
                cur_event.exec_num = item.second.size() / loops;
                double event_exec_time = avg_exec_time * double(cur_event.exec_num);
                cur_event.percentage_of_op = event_exec_time / avg_external_time * 100.0;
                total_kernel_time += event_exec_time;
                total_percentable_of_op += cur_event.percentage_of_op;
                if (cur_event.event_name.rfind("Rns", 0) == 0){
                    total_NTT_time += event_exec_time;
                    total_percentable_of_NTT += cur_event.percentage_of_op;
                }
                event_stats.push_back(cur_event);
            }
            
            export_table_ += "Total\t" + std::to_string(total_kernel_time) + "\t\t\t" + std::to_string(total_percentable_of_op) + "\n";
            if (total_NTT_time > 1e-6){
                export_table_ += "\t\tTotal NTT\t" + std::to_string(total_NTT_time) + "\t\t\t" + std::to_string(total_percentable_of_NTT) + "\n";
            }

            std::sort(event_stats.begin(), event_stats.end(), 
                      [](const EventStats &x, const EventStats &y){return x.percentage_of_op > y.percentage_of_op;});
        
            for (auto const& event_stat : event_stats){
                export_table_ += "\t\t";
                export_table_ += event_stat.event_name + "\t";
                export_table_ += std::to_string(event_stat.avg_time) + "\t";
                export_table_ += std::to_string(event_stat.exec_num) + "\t";
                double percentage_of_kernel = 100.0 * event_stat.avg_time * double(event_stat.exec_num) / total_kernel_time;
                export_table_ += std::to_string(percentage_of_kernel) + "\n";
            }
        }

        // export the formated string to a txt file
        static void export_table(std::string file_name){
            if (!activated_) return;
            std::ofstream f;
            f.open(file_name);
            f << export_table_;
            f.close();
        }

        // clear the formated string (export_table_)
        static void clear_export_table(){
            if (!activated_) return;
            export_table_ = "";
        }

    private:
        // map data structure used to record the submitted events
        inline static std::unordered_map<std::string, std::vector<cl::sycl::event>> events_;
        // formated string for record the export table
        inline static std::string export_table_;
        // current switch for activating/disabling EventCollector
        inline static bool activated_ = true;
    }; 

    template<class T>
    void gpu_copy(cl::sycl::queue& q, T* dst, const T* src, size_t len, bool wait = true)
    {
#if 0
        std::memcpy(dst, src, sizeof(T) * len);
#else
        if (len > 0)
        {
            auto e = q.submit([&](cl::sycl::handler& h)
            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
            {
                // copy
                h.memcpy(dst, src, sizeof(T) * len);
            });
            EventCollector::add_event("gpu_copy", e);

            if (wait){
                q.wait();
            }
        }
#endif
    }

    /*
      Theoretically this is the only place where wait has to be called to sync with host
    */
    template<class T>
    void gpu_host_get(cl::sycl::queue& q, T* dst, const T* src, size_t len, bool wait = true)
    {
        if (len > 0)
        {

            auto e= q.submit([&](cl::sycl::handler& h)
            [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
            {
                // copy
                h.memcpy(dst, src, sizeof(T) * len);
            });
            EventCollector::add_event("gpu_host_get", e);

            if (wait){
                q.wait();
            }
        }

    }

    template<class T>
    void gpu_set(cl::sycl::queue& q, T* dst, T value, size_t len, bool wait = true)
    {
#if 0
        for (size_t i = 0; i < len; ++i)
        {
            dst[i] = value;
        }
#else

        if (len > 0)
        {
            if constexpr (sizeof(T) == 4)
            {
                // value is not templated
                auto e = q.submit([&](cl::sycl::handler& h)
                [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                {
                    h.memset(dst, value, sizeof(T) * len);
                });
                EventCollector::add_event("gpu_set", e);
            }
            else
            {
                auto e = q.submit([&](cl::sycl::handler& h) {
                    h.parallel_for(len, [=](cl::sycl::id<1> i)
                    [[intel::reqd_sub_group_size(_SIMD_WIDTH_)]]
                    {
                        dst[i] = value;
                        });
                    });
                EventCollector::add_event("gpu_set", e);
            }

            if (wait){
                q.wait();
            }
        }
#endif 
    }
    
#endif // #if defined(BUILD_WITH_IGPU) || defined(SEAL_USE_INTEL_GPU)

} // namespace xehe

#define XEHE_READ_W64(IN_U64, IDX, STRIDE, VAR_W64)\
          { \
              VAR_W64.part[0] = ((uint32_t*)&IN_U64[(IDX)])[0];\
              VAR_W64.part[1] = ((uint32_t*)&IN_U64[(IDX)])[1]; \
          }

#define XEHE_WRITE_W64(OUT_U64, IDX, STRIDE, VAR_W64)\
          { \
              ((uint32_t*)&OUT_U64[(IDX)])[0] = VAR_W64.part[0]; \
              ((uint32_t*)&OUT_U64[(IDX)])[1] = VAR_W64.part[1]; \
          }

#define XEHE_WRITE_W128(OUT_U64, IDX, STRIDE, VAR_W128)\
          { \
              ((uint32_t*)&OUT_U64[(IDX)*2])[0] = VAR_W128.part[0].part[0]; \
              ((uint32_t*)&OUT_U64[(IDX)*2])[1] = VAR_W128.part[0].part[1]; \
              ((uint32_t*)&OUT_U64[(IDX)*2])[2] = VAR_W128.part[1].part[0]; \
              ((uint32_t*)&OUT_U64[(IDX)*2])[3] = VAR_W128.part[1].part[1]; \
          }





#endif //XeHE_LIB_UTILS_H
