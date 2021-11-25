
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

import dpctl


def create_default_queue():
    """Create a queue from default selector."""
    q = dpctl.SyclQueue()
    # Queue is out-of-order by default
    print("Queue {} is in order: {}".format(q, q.is_in_order))


def create_queue_from_filter_selector():
    """Create queue for a GPU device or,
    if it is not available, for a CPU device.
    Create in-order queue with profilign enabled.
    """
    q = dpctl.SyclQueue("gpu,cpu", property=("in_order", "enable_profiling"))
    print("Queue {} is in order: {}".format(q, q.is_in_order))
    # display the device used
    print("Device targeted by the queue:")
    q.sycl_device.print_device_info()


def create_queue_from_device():
    """
    Create a queue from SyclDevice instance.
    """
    cpu_d = dpctl.SyclDevice("opencl:cpu:0")
    q = dpctl.SyclQueue(cpu_d, property="enable_profiling")
    assert q.sycl_device == cpu_d
    print(
        "Number of devices in SyclContext " "associated with the queue: ",
        q.sycl_context.device_count,
    )


def create_queue_from_subdevice():
    """
    Create a queue from a sub-device.
    """
    cpu_d = dpctl.SyclDevice("opencl:cpu:0")
    sub_devs = cpu_d.create_sub_devices(partition=4)
    q = dpctl.SyclQueue(sub_devs[0])
    # a single-device context is created automatically
    print(
        "Number of devices in SyclContext " "associated with the queue: ",
        q.sycl_context.device_count,
    )


def create_queue_from_subdevice_multidevice_context():
    """
    Create a queue from a sub-device.
    """
    cpu_d = dpctl.SyclDevice("opencl:cpu:0")
    sub_devs = cpu_d.create_sub_devices(partition=4)
    ctx = dpctl.SyclContext(sub_devs)
    q = dpctl.SyclQueue(ctx, sub_devs[0])
    # a single-device context is created automatically
    print(
        "Number of devices in SyclContext " "associated with the queue: ",
        q.sycl_context.device_count,
    )


if __name__ == "__main__":
    create_default_queue()
    create_queue_from_filter_selector()
    create_queue_from_device()
    create_queue_from_subdevice()
    create_queue_from_subdevice_multidevice_context()   
