
#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Examples illustrating SYCL device selection features provided by dpctl.
"""

import dpctl


def print_device(d):
    "Display information about given device argument."
    if type(d) is not dpctl.SyclDevice:
        raise ValueError
    print("Name: ", d.name)
    print("Vendor: ", d.vendor)
    print("Driver version: ", d.driver_version)
    print("Backend: ", d.backend)
    print("Max EU: ", d.max_compute_units)


def create_default_device():
    """
    Create default SyclDevice using `cl::sycl::default_selector`.
    Device created can be influenced by environment variable
    SYCL_DEVICE_FILTER, which determines SYCL devices seen by the
    SYCL runtime.
    """
    d1 = dpctl.SyclDevice()
    d2 = dpctl.select_default_device()
    assert d1 == d2
    print_device(d1)
    return d1


def create_gpu_device():
    """
    Create a GPU device.
    Device created can be influenced by environment variable
    SYCL_DEVICE_FILTER, which determines SYCL devices seen by the
    SYCL runtime.
    """
    d1 = dpctl.SyclDevice("gpu")
    d2 = dpctl.select_gpu_device()
    assert d1 == d2
    print_device(d1)
    return d1


def create_gpu_device_if_present():
    """
    Select from union of two selections using default_selector.
    If a GPU device is available, it will be selected, if not,
    a CPU device will be selected, if available, otherwise an error
    will be raised.
    Device created can be influenced by environment variable
    SYCL_DEVICE_FILTER, which determines SYCL devices seen by the
    SYCL runtime.
    """
    d = dpctl.SyclDevice("gpu,cpu")
    print("Selected " + ("GPU" if d.is_gpu else "CPU") + " device")


def custom_select_device():
    """
    Programmatically select among available devices.
    Device created can be influenced by environment variable
    SYCL_DEVICE_FILTER, which determines SYCL devices seen by the
    SYCL runtime.
    """
    # select devices that support half-precision computation
    devs = [d for d in dpctl.get_devices() if d.has_aspect_fp16]
    # choose the device with highest default_selector score
    max_score = 0
    selected_dev = None
    for d in devs:
        if d.default_selector_score > max_score:
            max_score = d.default_selector_score
            selected_dev = d
    if selected_dev:
        print_device(selected_dev)
    else:
        print("No device with half-precision support is available.")
    return selected_dev


if __name__ == "__main__":
    create_default_device()
    create_gpu_device()
    create_gpu_device_if_present()
    custom_select_device()
