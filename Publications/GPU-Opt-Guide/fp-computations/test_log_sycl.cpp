//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
#include <CL/sycl.hpp>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <cmath>

#if FP_SIZE == 32
  typedef float FP_TYPE;
  static constexpr FP_TYPE VALIDATION_THRESHOLD = 1e-3;
#elif FP_SIZE == 64
  typedef double FP_TYPE;
  static constexpr FP_TYPE VALIDATION_THRESHOLD = 1e-6;
#endif

template<typename T>
void do_work_std (sycl::queue &q, unsigned NELEMENTS, unsigned NREPETITIONS, T initial_value, T *res)
{
	q.submit([&](sycl::handler& h) {
		h.parallel_for(NELEMENTS, [=] (auto j)
		{
			FP_TYPE tmp = initial_value;
			for (unsigned i = 0; i < NREPETITIONS; ++i)
				tmp += std::log(tmp);
			res[j] = tmp;
			});
	}).wait();
}

template<typename T>
void do_work_sycl (sycl::queue &q, unsigned NELEMENTS, unsigned NREPETITIONS, T initial_value, T *res)
{
	q.submit([&](sycl::handler& h) {
		h.parallel_for(NELEMENTS, [=] (auto j)
		{
			FP_TYPE tmp = initial_value;
			for (unsigned i = 0; i < NREPETITIONS; ++i)
				tmp += sycl::log(tmp);
			res[j] = tmp;
			});
	}).wait();
}
# if FP_SIZE == 32
template<typename T>
void do_work_sycl_native (sycl::queue &q, unsigned NELEMENTS, unsigned NREPETITIONS, T initial_value, T *res)
{
	q.submit([&](sycl::handler& h) {
		h.parallel_for(NELEMENTS, [=] (auto j)
		{
			FP_TYPE tmp = initial_value;
			for (unsigned i = 0; i < NREPETITIONS; ++i)
				tmp += sycl::native::log(tmp);
			res[j] = tmp;
			});
	}).wait();
}
# endif

int main (int argc, char *argv[])
{
	static constexpr unsigned NELEMENTS = 64*1024*1024;
	static constexpr unsigned NREPETITIONS = 1024;

	sycl::device d (sycl::gpu_selector_v);
	sycl::queue q (d);

	q.submit([&](sycl::handler& h) {
		h.single_task ([=]() { });
	}).wait();

	FP_TYPE initial_value = 2;
	FP_TYPE ref_res = initial_value;
	for (unsigned i = 0; i < NREPETITIONS; ++i)
		ref_res += std::log(ref_res);
	std::cout << "reference result = " << ref_res << std::endl;

	{
		FP_TYPE * std_res = new FP_TYPE[NELEMENTS];
		assert (std_res != nullptr);

		std::chrono::duration<float, std::micro> elapsed;

		{
			auto * res = sycl::malloc_device<FP_TYPE>(NELEMENTS, q);
			auto tbegin = std::chrono::system_clock::now();
			do_work_std<FP_TYPE>(q, NELEMENTS, NREPETITIONS, initial_value, res);
			auto tend = std::chrono::system_clock::now();
			elapsed = tend - tbegin;
			q.memcpy (std_res, res, NELEMENTS*sizeof(FP_TYPE)).wait();
			sycl::free (res, q);
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

	{
		FP_TYPE * sycl_res = new FP_TYPE[NELEMENTS];
		assert (sycl_res != nullptr);

		std::chrono::duration<float, std::micro> elapsed;

		{
			auto * res = sycl::malloc_device<FP_TYPE>(NELEMENTS, q);
			auto tbegin = std::chrono::system_clock::now();
			do_work_sycl<FP_TYPE>(q, NELEMENTS, NREPETITIONS, initial_value, res);
			auto tend = std::chrono::system_clock::now();
			elapsed = tend - tbegin;
			q.memcpy (sycl_res, res, NELEMENTS*sizeof(FP_TYPE)).wait();
			sycl::free (res, q);
		}
		std::cout << "sycl::log result[0] = " << sycl_res[0] << std::endl;

		bool allequal = true;
		for (auto i = 1; i < NELEMENTS; ++i)
			allequal = allequal and sycl_res[0] == sycl_res[i];
		if (allequal)
		{
			if (std::abs(ref_res - sycl_res[0])/std::abs(ref_res) < std::abs(VALIDATION_THRESHOLD))
				std::cout << "sycl::log validates. Total execution time is " << elapsed.count() << " us." << std::endl;
			else
				std::cout << "sycl::log does not validate (ref=" << ref_res << " sycl_res=" << sycl_res[0] << " mix=" << std::abs(ref_res - sycl_res[0])/std::abs(ref_res) << ")" << std::endl;
		}
		else
			std::cout << "sycl::log does not validate, results are not equal." << std::endl;

		delete [] sycl_res;
	}
# if FP_SIZE == 32
	{
		FP_TYPE * sycl_native_res = new FP_TYPE[NELEMENTS];
		assert (sycl_native_res != nullptr);

		std::chrono::duration<float, std::micro> elapsed;

		{
			auto * res = sycl::malloc_device<FP_TYPE>(NELEMENTS, q);
			auto tbegin = std::chrono::system_clock::now();
			do_work_sycl_native<FP_TYPE>(q, NELEMENTS, NREPETITIONS, initial_value, res);
			auto tend = std::chrono::system_clock::now();
			elapsed = tend - tbegin;
			q.memcpy (sycl_native_res, res, NELEMENTS*sizeof(FP_TYPE)).wait();
			sycl::free (res, q);
		}
		std::cout << "sycl::native::log result[0] = " << sycl_native_res[0] << std::endl;

		bool allequal = true;
		for (auto i = 1; i < NELEMENTS; ++i)
			allequal = allequal and sycl_native_res[0] == sycl_native_res[i];
		if (allequal)
		{
			if (std::abs(ref_res - sycl_native_res[0])/std::abs(ref_res) < std::abs(VALIDATION_THRESHOLD))
				std::cout << "sycl::native::log validates. Total execution time is " << elapsed.count() << " us." << std::endl;
			else
				std::cout << "sycl::native::log does not validate (ref=" << ref_res << " sycl_native_res=" << sycl_native_res[0] << " mix=" << std::abs(ref_res - sycl_native_res[0])/std::abs(ref_res) << ")" << std::endl;
		}
		else
			std::cout << "sycl::native::log does not validate, results are not equal." << std::endl;

		delete [] sycl_native_res;
	}
# endif // FP_SIZE == 32

	return 0;
}
