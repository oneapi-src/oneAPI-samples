// Some Halide runtime functions adapted to use in Sycl.

#include <sycl/sycl.hpp>
#include "halide_runtime_etc.hpp"

void halide_sycl_device_malloc(struct halide_buffer_t *buf, const sycl::queue &q_device) {
    // TOFIX: recover this assertion. Currently, the compiler might generate device alloc more than once.
    //assert(!buf->device);
    if (!buf->device) {
        assert(buf->size_in_bytes() != 0);
        device_handle *dev_handle = (device_handle *)std::malloc(sizeof(device_handle));
        dev_handle->mem = (void*)sycl::malloc_device(buf->size_in_bytes(), q_device);
        dev_handle->offset = 0;
        buf->device = (uint64_t)dev_handle;
    }
}

void halide_sycl_host_malloc(struct halide_buffer_t *buf) {
    assert(buf->size_in_bytes() != 0);
    buf->host = (uint8_t*)std::malloc(buf->size_in_bytes());
    assert(buf->host != NULL);
}

void halide_sycl_device_and_host_malloc(struct halide_buffer_t *buf, const sycl::queue &q_device) {
    halide_sycl_device_malloc(buf, q_device);
    halide_sycl_host_malloc(buf);
}

void halide_sycl_device_free(halide_buffer_t *buf, const sycl::queue &q_device) {
    device_handle *dev_handle = (device_handle *)buf->device;
    if (dev_handle) {
        sycl::free(dev_handle->mem, q_device);
        assert(dev_handle->offset == 0);

        std::free(dev_handle);
        buf->device = 0;
        buf->set_device_dirty(false);
    }
}

void halide_sycl_host_free(halide_buffer_t *buf) {
    if (buf->host) {
        std::free((void*)buf->host);
        buf->host = NULL;
        buf->set_host_dirty(false);
    };
}

void halide_sycl_device_and_host_free(halide_buffer_t *buf, const sycl::queue &q_device) {
    halide_sycl_device_free(buf, q_device);
    halide_sycl_host_free(buf);
}


void halide_sycl_buffer_copy(halide_buffer_t *buf, bool to_host, sycl::queue &q_device) {
    bool from_host = (buf->device == 0) || (buf->host_dirty() && buf->host != NULL);
    if (!from_host && to_host) {
        // Device->host
        q_device.submit([&](sycl::handler& h){ h.memcpy((void *)buf->host, (void *)(((device_handle*)buf->device)->mem), buf->size_in_bytes()); }).wait();
    } else if (from_host && !to_host) {
        // Host->device
        q_device.submit([&](sycl::handler& h){ h.memcpy((void *)(((device_handle*)buf->device)->mem), (void *)buf->host, buf->size_in_bytes()); }).wait();
    } else if (!from_host && !to_host) {
        // Device->device: not supported
        assert(false);
    } else {
        // Host->host: Do nothing
    }
}
