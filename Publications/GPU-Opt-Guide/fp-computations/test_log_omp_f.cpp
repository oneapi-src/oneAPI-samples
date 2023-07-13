//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
#include <iostream>
#include <assert.h>
#include <chrono>
#include <cmath>

#if FP_SIZE == 32
  typedef float FP_TYPE;
  static constexpr FP_TYPE VALIDATION_THRESHOLD = 1e-3;
  extern "C" void log_real_sp (int, int, FP_TYPE, FP_TYPE *);

#elif FP_SIZE == 64
  typedef double FP_TYPE;
  static constexpr FP_TYPE VALIDATION_THRESHOLD = 1e-6;
  extern "C" void log_real_dp (int, int, FP_TYPE, FP_TYPE *);
#endif

int main (int argc, char *argv[])
{
	static constexpr unsigned NELEMENTS = 64*1024*1024;
	static constexpr int NREPETITIONS = 1024;

	#pragma omp target
	{ }

	FP_TYPE initial_value = 2;
	FP_TYPE ref_res = initial_value;
	for (unsigned i = 0; i < NREPETITIONS; ++i)
		ref_res += std::log(ref_res);
	std::cout << "reference result = " << ref_res << std::endl;

	{
		FP_TYPE * std_res = new FP_TYPE[NELEMENTS];
		assert (std_res != nullptr);

		std::chrono::duration<float, std::micro> elapsed;
		#pragma omp target data map(std_res[0:NELEMENTS])
		{
			auto tbegin = std::chrono::system_clock::now();
			{
# if FP_SIZE == 32
				log_real_sp (NELEMENTS, NREPETITIONS, initial_value, std_res);
# elif FP_SIZE == 64
				log_real_dp (NELEMENTS, NREPETITIONS, initial_value, std_res);
# endif

			}
			auto tend = std::chrono::system_clock::now();
			elapsed = tend - tbegin;
		}
		std::cout << "std::log result[0] = " << std_res[0] << std::endl;

		bool allequal = true;
		for (auto i = 1; i < NELEMENTS; ++i)
			allequal = allequal and std_res[0] == std_res[i];
		if (allequal)
		{
			if (std::abs(ref_res - std_res[0])/std::abs(ref_res) < std::abs(VALIDATION_THRESHOLD))
				std::cout << "std::log validates. Total execution time is " << elapsed.count() << " us." << std::endl;
			else
				std::cout << "std::log does not validate (ref=" << ref_res << " std_res=" << std_res[0] << " mix=" << std::abs(ref_res - std_res[0])/std::abs(ref_res) << ")" << std::endl;
		}
		else
			std::cout << "std::log does not validate, results are not equal." << std::endl;

		delete [] std_res;
	}

	return 0;
}
