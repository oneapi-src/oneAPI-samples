/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 * Copyright (c) 2019-2021 Intel Corporation (oneAPI modifications)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <CL/sycl.hpp>
#include <cstdint>

namespace pointpillars {

// Prefix sum in 2D coordinates
//
// These functions calculate the cumulative sum along X or Y in a 2D array
//
//          X--->
//            W
//  Y    o-------------
//  |    |
//  |  H |
//  v    |
//       |
//
// For details about the algorithm please check:
//   Sengupta, Shubhabrata & Lefohn, Aaron & Owens, John. (2006). A Work-Efficient Step-Efficient Prefix Sum Algorithm.
//

// Prefix in x-direction, calculates the cumulative sum along x
void ScanX(int *dev_output, const int *dev_input, int w, int h, int n);

// Prefix in y-direction, calculates the cumulative sum along y
void ScanY(int *dev_output, const int *dev_input, int w, int h, int n);

}  // namespace pointpillars
