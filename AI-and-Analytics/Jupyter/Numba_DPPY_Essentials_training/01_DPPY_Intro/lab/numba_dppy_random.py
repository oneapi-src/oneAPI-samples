#! /usr/bin/env python
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numba
import numpy as np
import dpctl


@numba.njit
def rand():
    return np.random.rand(3, 2)


@numba.njit
def random_sample(size):
    return np.random.random_sample(size)


@numba.njit
def random_exponential(scale, size):
    return np.random.exponential(scale, size)


@numba.njit
def random_normal(loc, scale, size):
    return np.random.normal(loc, scale, size)


size = 9
scale = 3.0

with dpctl.device_context("opencl:gpu"):
    result = rand()
    # Random values in a given shape (3, 2)
    print(result)

    result = random_sample(size)
    # Array of shape (9,) with random floats in the half-open interval [0.0, 1.0)
    print(result)

    result = random_exponential(scale, size)
    # Array of shape (9,) with samples from an exponential distribution
    print(result)

    result = random_normal(0.0, 0.1, size)
    # Array of shape (9,) with samples from a normal distribution
    print(result)
