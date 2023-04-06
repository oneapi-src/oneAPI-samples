/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#ifndef SANITY_CHECK_GPU_HPP
#define SANITY_CHECK_GPU_HPP

#include <array>
#include <iostream>
#include <iso646.h>
#include <random>


#include "../../../include/XeHE.hpp"


namespace xehe
{
	namespace util
	{
		template<class T>
		void XeHE_sanity_check(size_t data_len)
		{

			std::vector<T> host_vec1(data_len);
			auto host_ptr1 = host_vec1.data();

			for (size_t i = 0; i < data_len; ++i)
			{
				host_ptr1[i] = i;
			}

			auto buf1 = xehe::ext::XeHE_malloc<T>(data_len);
			buf1->set_data(host_ptr1);
			auto buf2 = xehe::ext::XeHE_malloc<T>(data_len);
			buf2->deep_copy(buf1);

			std::vector<T> host_vec2(data_len);
			auto host_ptr2 = host_vec2.data();

			buf2->get_data(host_ptr2);

			bool success = true;
			size_t fail_idx = 0;
			for (size_t i = 0; i < data_len && success; ++i)
			{
				if (host_ptr1[i] != host_ptr2[i])
				{
					success = false;
					fail_idx = i;
				}
			}
			if (!success)
			{
				std::cout << "Sanity check failed at " << fail_idx << " with " << host_ptr1[fail_idx] << " != " << host_ptr2[fail_idx] << std::endl;
			}
			else
			{
				std::cout << ((sizeof(T) == 8) ? "64bit " : "32bit ") << "sanity check is an enormous success!" << std::endl;
			}

		}
	}
}

#endif