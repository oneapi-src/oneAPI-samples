/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#include "perf_app.hpp"

/*
* Performance numbers in clc
* Add
* AddMod
* Mul
* MulMod
* Mulx2
* Mulx2Mod
*/


int main(int argc, char* argv[]){
    //xehe::dpcpp::Context ctx;
    //std::cout << "Hello with DPC++! Context created: " << &ctx << std::endl;


    // 32-bit - 2times more work
    perf_util_result result_32 = PerfUtil<uint32_t>(1, 1, 128, 16, 30);
 
    //
    perf_util_result result_64 = PerfUtil<uint64_t>(1, 614, 64, 16, 60);

    //printing the results
    std::cout << std::endl << std::endl << "||" << std::setfill('=') << std::setw(50);
    std::cout << " Perf Utility ";
    std::cout << std::setfill('=') << std::setw(36) << "||" << std::endl;
    
    //calibrated data
    std::cout << std::endl << std::setfill(' ') << std::setw(52);
    std::cout << "Calibrated Data";
    std::cout << std::setfill(' ') << std::setw(50) << std::endl << std::endl;

    std::cout << std::left << std::setw(30) << " " << std::left << std::setw(20) << "INT32" << std::left << std::setw(20) << "INT64" 
        << std::left << std::setw(20) << "% Change (INT32 -> INT64)" << std:: endl;
    
    std::cout << std::left << std::setw(30) << "n_EUs" << std::left << std::setw(20) << result_32.n_EUs << std::left << std::setw(20) 
        << result_64.n_EUs << std::left << std::setw(20) << std::setprecision(3) <<  (double)((result_64.n_EUs-result_32.n_EUs)/result_32.n_EUs)*100 << std::endl;
    
    std::cout << std::left << std::setw(30) << "Spec max engine freq" << std::left << std::setw(20) << result_32.spec_max_eng_freq << std::left << std::setw(20) 
        << result_64.spec_max_eng_freq << std::left << std::setw(20) << std::setprecision(3) << (double)((result_64.spec_max_eng_freq-result_32.spec_max_eng_freq)/result_32.spec_max_eng_freq)*100 << std::endl;
    
    std::cout << std::left << std::setw(30) << "Actual engine freq" << std::left << std::setw(20) << result_32.actual_eng_freq << std::left << std::setw(20) 
        << result_64.actual_eng_freq << std::left << std::setw(20) << std::setprecision(3) << (double)((result_64.actual_eng_freq-result_32.actual_eng_freq)/result_32.actual_eng_freq)*100 << std::endl;
    
    std::cout << std::left << std::setw(30) << "Duration" << std::left << std::setw(20) << result_32.duration << std::left << std::setw(20)
        << result_64.duration << std::left << std::setw(20) << std::setprecision(3) << (double)((result_64.duration-result_32.duration)/result_32.duration)*100 << std::endl;
    
    std::cout << std::left << std::setw(30) << "# instructions" << std::left << std::setw(20) << result_32.n_calib_instruction << std::left << std::setw(20) 
        << result_64.n_calib_instruction << std::left << std::setw(20) << std::setprecision(3) << (double)((result_64.n_calib_instruction-result_32.n_calib_instruction)/result_32.n_calib_instruction)*100 << std::endl;
    
    std::cout << std::left << std::setw(30) << "clc" << std::left << std::setw(20) << result_32.total_clc << std::left << std::setw(20) 
        << result_64.total_clc << std::left << std::setw(20) << std::setprecision(3) << (double)((result_64.total_clc-result_32.total_clc)/result_32.total_clc)*100 << std::endl;
    
    //instruction data
    std::cout << std::endl;
    std::cout << std::setfill(' ') << std::setw(52);
    std::cout << "Instruction's Data";
    std::cout << std::setfill(' ') << std::setw(50) << std::endl << std::endl;
    std::cout << std::left << std::setw(20) << "Instruction" << std::left << std::setw(20) << "Criteria" 
        << std::left << std::setw(20) << "INT32" << std::left << std::setw(20) << "INT64" << std::left 
        << std::setw(20) << "% Change (INT32 -> INT64)" << std::endl;
    
    for (int i = 0; i < inst_names.size(); i++){
        
        std::cout << std::left << std::setw(20) << inst_names_enh[i] << std::left << std::setw(20) << "Duration" 
            << std::left << std::setw(20) << result_32.insts[i].duration << std::left << std::setw(20) 
            << result_64.insts[i].duration << std::left << std::setw(20)
            << (double)((result_64.insts[i].duration-result_32.insts[i].duration)/result_32.insts[i].duration)*100 << std::endl;

        std::cout << std::left << std::setw(20) << " " << std::left << std::setw(20) << "clc" 
            << std::left << std::setw(20) << result_32.insts[i].clc << std::left << std::setw(20)
            << result_64.insts[i].clc << std::left << std::setw(20)
            << (double)((result_64.insts[i].clc-result_32.insts[i].clc)/result_32.insts[i].clc)*100 << std::endl;
    }
}
