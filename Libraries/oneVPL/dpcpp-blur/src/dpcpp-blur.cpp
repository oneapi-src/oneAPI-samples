//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// Minimal oneAPI Video Processing Library (oneVPL) dpc++ interop application.
///
/// @file

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include "vpl/mfxvideo.h"

#ifdef BUILD_DPCPP
    #include "CL/sycl.hpp"
#endif

#define MAX_PATH   260
#define MAX_WIDTH  3840
#define MAX_HEIGHT 2160

#define OUTPUT_FILE "out.rgba"

#define BLUR_RADIUS 5
#define BLUR_SIZE   (float)((BLUR_RADIUS << 1) + 1)

#ifdef BUILD_DPCPP
namespace dpc_common {
// this exception handler with catch async exceptions
static auto exception_handler = [](cl::sycl::exception_list exception_list) {
    for (std::exception_ptr const &e : exception_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const &e) {
    #if _DEBUG
            std::cout << "Failure" << std::endl;
    #endif
            std::terminate();
        }
    }
};
}; // namespace dpc_common

// Select device on which to run kernel.
class MyDeviceSelector : public cl::sycl::device_selector {
public:
    MyDeviceSelector() {}

    int operator()(const cl::sycl::device &device) const override {
        const std::string name =
            device.get_info<cl::sycl::info::device::name>();

        std::cout << "Trying device: " << name << "..." << std::endl;
        std::cout << "  Vendor: "
                  << device.get_info<cl::sycl::info::device::vendor>()
                  << std::endl;

        if (device.is_cpu())
            return 500; // We give higher merit for CPU
        //if (device.is_accelerator()) return 400;
        //if (device.is_gpu()) return 300;
        //if (device.is_host()) return 100;
        return -1;
    }
};

void BlurFrame(sycl::queue q,
               mfxFrameSurface1 *in_surface,
               mfxFrameSurface1 *blured_surface);
#endif

mfxStatus LoadRawFrame(mfxFrameSurface1 *surface, FILE *f);
void WriteRawFrame(mfxFrameSurface1 *surface, FILE *f);
mfxU32 GetSurfaceSize(mfxU32 fourcc, mfxU32 width, mfxU32 height);
mfxI32 GetFreeSurfaceIndex(const std::vector<mfxFrameSurface1> &surface_pool);
char *ValidateFileName(char *in);

// Print usage message
void Usage(void) {
    printf("Usage: dpcpp-blur SOURCE WIDTH HEIGHT\n\n"
           "Process raw I420 video in SOURCE having dimensions WIDTH x HEIGHT "
           "to blurred raw RGB32 video in %s\n"
           "Default blur kernel is [%d]x[%d]\n\n"
           "To view:\n"
           " ffplay -video_size [width]x[height] "
           "-pixel_format rgb32 -f rawvideo %s\n",
           OUTPUT_FILE,
           2 * BLUR_RADIUS + 1,
           2 * BLUR_RADIUS + 1,
           OUTPUT_FILE);
}

int main(int argc, char *argv[]) {
    mfxU32 fourcc;
    mfxU32 width;
    mfxU32 height;

    if (argc != 4) {
        Usage();
        return 1;
    }

    char *in_filename = NULL;

    in_filename = ValidateFileName(argv[1]);
    if (!in_filename) {
        printf("Input filename is not valid\n");
        Usage();
        return 1;
    }

    FILE *source = fopen(in_filename, "rb");
    if (!source) {
        printf("Could not open input file, \"%s\"\n", in_filename);
        return 1;
    }
    FILE *sink = fopen(OUTPUT_FILE, "wb");
    if (!sink) {
        fclose(source);
        printf("Could not open output file, %s\n", OUTPUT_FILE);
        return 1;
    }
    mfxI32 isize = strtol(argv[2], NULL, 10);
    if (isize <= 0 || isize > MAX_WIDTH) {
        fclose(source);
        fclose(sink);
        puts("Input size is not valid\n");
        return 1;
    }
    mfxI32 input_width = isize;

    isize = strtol(argv[3], NULL, 10);
    if (isize <= 0 || isize > MAX_HEIGHT) {
        fclose(source);
        fclose(sink);
        puts("Input size is not valid\n");
        return 1;
    }
    mfxI32 input_height = isize;

    // initialize  session
    mfxInitParam init_params   = { 0 };
    init_params.Version.Major  = 2;
    init_params.Version.Minor  = 0;
    init_params.Implementation = MFX_IMPL_SOFTWARE;

    mfxSession session;
    mfxStatus sts = MFXInitEx(init_params, &session);
    if (sts != MFX_ERR_NONE) {
        fclose(source);
        fclose(sink);
        puts("MFXInitEx error.  Could not initialize session");
        return 1;
    }

    // Initialize VPP parameters

    // - For simplistic memory management, system memory surfaces are used to
    //   store the raw frames (Note that when using HW acceleration video
    //   surfaces are prefered, for better performance)
    mfxVideoParam vpp_params;
    memset(&vpp_params, 0, sizeof(vpp_params));
    // Input data
    vpp_params.vpp.In.FourCC        = MFX_FOURCC_I420;
    vpp_params.vpp.In.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    vpp_params.vpp.In.CropX         = 0;
    vpp_params.vpp.In.CropY         = 0;
    vpp_params.vpp.In.CropW         = input_width;
    vpp_params.vpp.In.CropH         = input_height;
    vpp_params.vpp.In.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    vpp_params.vpp.In.FrameRateExtN = 30;
    vpp_params.vpp.In.FrameRateExtD = 1;
    vpp_params.vpp.In.Width         = vpp_params.vpp.In.CropW;
    vpp_params.vpp.In.Height        = vpp_params.vpp.In.CropH;
    // Output data in the same size but with RGB32 color format
    vpp_params.vpp.Out.FourCC        = MFX_FOURCC_RGB4;
    vpp_params.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    vpp_params.vpp.Out.CropX         = 0;
    vpp_params.vpp.Out.CropY         = 0;
    vpp_params.vpp.Out.CropW         = input_width;
    vpp_params.vpp.Out.CropH         = input_height;
    vpp_params.vpp.Out.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    vpp_params.vpp.Out.FrameRateExtN = 30;
    vpp_params.vpp.Out.FrameRateExtD = 1;
    vpp_params.vpp.Out.Width         = vpp_params.vpp.Out.CropW;
    vpp_params.vpp.Out.Height        = vpp_params.vpp.Out.CropH;

    vpp_params.IOPattern =
        MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

    // Query number of required surfaces for VPP
    mfxFrameAllocRequest vpp_request[2]; // [0] - in, [1] - out
    memset(&vpp_request, 0, sizeof(mfxFrameAllocRequest) * 2);
    sts = MFXVideoVPP_QueryIOSurf(session, &vpp_params, vpp_request);
    if (sts != MFX_ERR_NONE) {
        fclose(source);
        fclose(sink);
        puts("QueryIOSurf error");
        return 1;
    }

    mfxU16 num_vpp_surfaces_in  = vpp_request[0].NumFrameSuggested;
    mfxU16 num_vpp_surfaces_out = vpp_request[1].NumFrameSuggested;

    // Allocate surfaces for VPP: In

    // Frame surface array keeps pointers to all surface planes and general
    // frame info
    fourcc = vpp_params.vpp.In.FourCC;
    width  = vpp_params.vpp.In.Width;
    height = vpp_params.vpp.In.Height;

    mfxU32 surface_size = GetSurfaceSize(fourcc, width, height);
    if (surface_size == 0) {
        fclose(source);
        fclose(sink);
        puts("VPP-in surface size is wrong");
        return 1;
    }

    std::vector<mfxU8> surface_data_in(surface_size * num_vpp_surfaces_in);
    mfxU8 *surface_buffers_in = surface_data_in.data();

    std::vector<mfxFrameSurface1> vpp_surfaces_in(num_vpp_surfaces_in);
    for (mfxI32 i = 0; i < num_vpp_surfaces_in; i++) {
        memset(&vpp_surfaces_in[i], 0, sizeof(mfxFrameSurface1));
        vpp_surfaces_in[i].Info   = vpp_params.vpp.In;
        vpp_surfaces_in[i].Data.Y = &surface_buffers_in[surface_size * i];
        vpp_surfaces_in[i].Data.U = vpp_surfaces_in[i].Data.Y + width * height;
        vpp_surfaces_in[i].Data.V =
            vpp_surfaces_in[i].Data.U + ((width / 2) * (height / 2));
        vpp_surfaces_in[i].Data.Pitch = width;
    }

    // Allocate surfaces for VPP: Out

    // Frame surface array keeps pointers to all surface planes and general
    // frame info
    fourcc = vpp_params.vpp.Out.FourCC;
    width  = vpp_params.vpp.Out.Width;
    height = vpp_params.vpp.Out.Height;

    surface_size = GetSurfaceSize(fourcc, width, height);
    if (surface_size == 0) {
        fclose(source);
        fclose(sink);
        puts("VPP-out surface size is wrong");
        return 1;
    }

    // we need one more surface for the blured image
    std::vector<mfxU8> surface_data_out(surface_size *
                                        (num_vpp_surfaces_out + 1));
    mfxU8 *surface_buffers_out = surface_data_out.data();

    std::vector<mfxFrameSurface1> vpp_surfaces_out(num_vpp_surfaces_out);
    for (mfxI32 i = 0; i < num_vpp_surfaces_out; i++) {
        memset(&vpp_surfaces_out[i], 0, sizeof(mfxFrameSurface1));
        vpp_surfaces_out[i].Info       = vpp_params.vpp.Out;
        vpp_surfaces_out[i].Data.B     = &surface_buffers_out[surface_size * i];
        vpp_surfaces_out[i].Data.G     = vpp_surfaces_out[i].Data.B + 1;
        vpp_surfaces_out[i].Data.R     = vpp_surfaces_out[i].Data.G + 1;
        vpp_surfaces_out[i].Data.A     = vpp_surfaces_out[i].Data.R + 1;
        vpp_surfaces_out[i].Data.Pitch = width * 4;
    }

    // Initialize surface for blured frame
    mfxFrameSurface1 blured_surface;
    std::vector<mfxU8> blur_data_out(surface_size);

    memset(&blured_surface, 1, sizeof(blured_surface));
    blured_surface.Info       = vpp_params.vpp.Out;
    blured_surface.Data.B     = &blur_data_out[0];
    blured_surface.Data.G     = blured_surface.Data.B + 1;
    blured_surface.Data.R     = blured_surface.Data.G + 1;
    blured_surface.Data.A     = blured_surface.Data.R + 1;
    blured_surface.Data.Pitch = width * 4;

    // Initialize VPP
    sts = MFXVideoVPP_Init(session, &vpp_params);
    if (sts != MFX_ERR_NONE) {
        fclose(source);
        fclose(sink);
        puts("Could not initialize vpp");
        return 1;
    }

    // Prepare bit stream buffer
    mfxBitstream bitstream = { 0 };
    bitstream.MaxLength    = 2000000;
    std::vector<mfxU8> bitstream_data(bitstream.MaxLength);
    bitstream.Data = bitstream_data.data();

    // Start processing the frames
    int surface_index_in = 0, surface_index_out = 0;
    mfxSyncPoint syncp;
    mfxU32 framenum = 0;

#ifdef BUILD_DPCPP
    // Initialize DPC++
    MyDeviceSelector sel;

    // Create SYCL execution queue
    sycl::queue q(sel, dpc_common::exception_handler);

    // See what device was actually selected for this queue.
    // CPU is preferrable for this time.
    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
#endif

    printf("Processing %s -> %s\n", in_filename, OUTPUT_FILE);

    // Stage 1: Main processing loop
    while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts) {
        surface_index_in =
            GetFreeSurfaceIndex(vpp_surfaces_in); // Find free frame surface
        if (surface_index_in == MFX_ERR_NOT_FOUND) {
            fclose(source);
            fclose(sink);
            puts("no available surface");
            return 1;
        }

        sts = LoadRawFrame(&vpp_surfaces_in[surface_index_in], source);
        if (sts != MFX_ERR_NONE)
            break;

        surface_index_out = GetFreeSurfaceIndex(
            vpp_surfaces_out); // Find free output frame surface
        if (surface_index_out == MFX_ERR_NOT_FOUND) {
            fclose(source);
            fclose(sink);
            puts("no available surface");
            return 1;
        }

        for (;;) {
            // Process a frame asychronously (returns immediately)
            sts = MFXVideoVPP_RunFrameVPPAsync(
                session,
                &vpp_surfaces_in[surface_index_in],
                &vpp_surfaces_out[surface_index_out],
                NULL,
                &syncp);

            if (MFX_ERR_NONE < sts && syncp) {
                sts = MFX_ERR_NONE; // Ignore warnings if output is available
                break;
            }
            else if (MFX_ERR_NOT_ENOUGH_BUFFER == sts) {
                // Allocate more bitstream buffer memory here if needed...
                break;
            }
            else {
                break;
            }
        }

        if (MFX_ERR_NONE == sts) {
            sts = MFXVideoCORE_SyncOperation(
                session,
                syncp,
                60000); // Synchronize. Wait until a frame is ready
            ++framenum;
            // Blur and store processed frame
#ifdef BUILD_DPCPP
            BlurFrame(q, &vpp_surfaces_out[surface_index_out], &blured_surface);
            WriteRawFrame(&blured_surface, sink);
#else
            WriteRawFrame(&vpp_surfaces_out[surface_index_out], sink);
#endif
            bitstream.DataLength = 0;
        }
    }

    sts = MFX_ERR_NONE;

    // Stage 2: Retrieve the buffered processed frames
    while (MFX_ERR_NONE <= sts) {
        surface_index_out =
            GetFreeSurfaceIndex(vpp_surfaces_out); // Find free frame surface
        if (surface_index_out == MFX_ERR_NOT_FOUND) {
            fclose(source);
            fclose(sink);
            puts("no available surface");
            return 1;
        }

        for (;;) {
            // Process a frame asychronously (returns immediately)
            sts = MFXVideoVPP_RunFrameVPPAsync(
                session,
                NULL,
                &vpp_surfaces_out[surface_index_out],
                NULL,
                &syncp);

            if (MFX_ERR_NONE < sts && syncp) {
                sts = MFX_ERR_NONE; // Ignore warnings if output is available
                break;
            }
            else {
                break;
            }
        }

        if (MFX_ERR_NONE == sts) {
            sts = MFXVideoCORE_SyncOperation(
                session,
                syncp,
                60000); // Synchronize. Wait until a frame is ready
            ++framenum;
            // Blur and store processed frame
#ifdef BUILD_DPCPP
            BlurFrame(q, &vpp_surfaces_out[surface_index_out], &blured_surface);
            WriteRawFrame(&blured_surface, sink);
#else
            WriteRawFrame(&vpp_surfaces_out[surface_index_out], sink);
#endif

            bitstream.DataLength = 0;
        }
    }

    printf("Processed %d frames\n", framenum);

    // Clean up resources - It is recommended to close components first, before
    // releasing allocated surfaces, since some surfaces may still be locked by
    // internal resources.
    MFXVideoVPP_Close(session);

    fclose(source);
    fclose(sink);

    return 0;
}

mfxStatus LoadRawFrame(mfxFrameSurface1 *surface, FILE *f) {
    mfxU16 w, h, i, pitch;
    mfxU32 bytes_read;
    mfxU8 *ptr;
    mfxFrameInfo *info = &surface->Info;
    mfxFrameData *data = &surface->Data;

    w = info->Width;
    h = info->Height;

    switch (info->FourCC) {
        case MFX_FOURCC_I420:
            // read luminance plane (Y)
            pitch = data->Pitch;
            ptr   = data->Y;
            for (i = 0; i < h; i++) {
                bytes_read = (mfxU32)fread(ptr + i * pitch, 1, w, f);
                if (w != bytes_read)
                    return MFX_ERR_MORE_DATA;
            }

            // read chrominance (U, V)
            pitch /= 2;
            h /= 2;
            w /= 2;
            ptr = data->U;
            for (i = 0; i < h; i++) {
                bytes_read = (mfxU32)fread(ptr + i * pitch, 1, w, f);
                if (w != bytes_read)
                    return MFX_ERR_MORE_DATA;
            }

            ptr = data->V;
            for (i = 0; i < h; i++) {
                bytes_read = (mfxU32)fread(ptr + i * pitch, 1, w, f);
                if (w != bytes_read)
                    return MFX_ERR_MORE_DATA;
            }
            break;
        default:
            break;
    }

    return MFX_ERR_NONE;
}

// Write raw rgb32 frame to file
void WriteRawFrame(mfxFrameSurface1 *surface, FILE *f) {
    mfxU16 w, h, i, pitch;
    mfxFrameInfo *info = &surface->Info;
    mfxFrameData *data = &surface->Data;

    w = info->Width;
    h = info->Height;

    // write the output to disk
    switch (info->FourCC) {
        case MFX_FOURCC_RGB4:
            pitch = data->Pitch;
            for (i = 0; i < h; i++) {
                fwrite(data->B + i * pitch, 1, w * 4, f);
            }
        default:
            break;
    }

    return;
}

// Return the surface size in bytes given format and dimensions
mfxU32 GetSurfaceSize(mfxU32 fourcc, mfxU32 width, mfxU32 height) {
    mfxU32 bytes = 0;

    switch (fourcc) {
        case MFX_FOURCC_I420:
            bytes = width * height + (width >> 1) * (height >> 1) +
                    (width >> 1) * (height >> 1);
            break;
        case MFX_FOURCC_RGB4:
            bytes = 4 * width * height;
            break;
        default:
            break;
    }

    return bytes;
}

// Return index of free surface in given pool
mfxI32 GetFreeSurfaceIndex(const std::vector<mfxFrameSurface1> &surface_pool) {
    auto it = std::find_if(surface_pool.begin(),
                           surface_pool.end(),
                           [](const mfxFrameSurface1 &surface) {
                               return 0 == surface.Data.Locked;
                           });

    if (it == surface_pool.end())
        return MFX_ERR_NOT_FOUND;
    else
        return static_cast<mfxI32>(it - surface_pool.begin());
}

char *ValidateFileName(char *in) {
    if (in) {
        if (strlen(in) > MAX_PATH)
            return NULL;
    }

    return in;
}

#ifdef BUILD_DPCPP

// Few useful acronyms.
constexpr auto sycl_read  = sycl::access::mode::read;
constexpr auto sycl_write = sycl::access::mode::write;

// SYCL kernel scheduler
// Blur frame by using SYCL kernel
void BlurFrame(sycl::queue q,
               mfxFrameSurface1 *in_surface,
               mfxFrameSurface1 *blured_surface) {
    int img_width, img_height;

    img_width  = in_surface->Info.Width;
    img_height = in_surface->Info.Height;

    // Wrap mfx surfaces into SYCL image by using host ptr for zero copy of data
    sycl::image<2> image_buf_src(in_surface->Data.B,
                                 sycl::image_channel_order::rgba,
                                 sycl::image_channel_type::unsigned_int8,
                                 sycl::range<2>(img_width, img_height));

    sycl::image<2> image_buf_dst(blured_surface->Data.B,
                                 sycl::image_channel_order::rgba,
                                 sycl::image_channel_type::unsigned_int8,
                                 sycl::range<2>(img_width, img_height));

    try {
        q.submit([&](cl::sycl::handler &cgh) {
            // Src image accessor
            sycl::accessor<cl::sycl::uint4,
                           2,
                           sycl_read,
                           sycl::access::target::image>
                accessor_src(image_buf_src, cgh);
            // Dst image accessor
            auto accessor_dst =
                image_buf_dst.get_access<cl::sycl::uint4, sycl_write>(cgh);
            cl::sycl::uint4 black = (cl::sycl::uint4)(0);
            // Parallel execution of the kerner for each pixel. Kernel
            // implemented as a lambda function.

            // Important: this is naive implementation of the blur kernel. For
            // further optimization it is better to use range_nd iterator and
            // apply moving average technique to reduce # of MAC operations per
            // pixel.
            cgh.parallel_for<class NaiveBlur_rgba>(
                sycl::range<2>(img_width, img_height),
                [=](sycl::item<2> item) {
                    auto coords = cl::sycl::int2(item[0], item[1]);

                    // Let's add horizontal black border
                    if (item[0] <= BLUR_RADIUS ||
                        item[0] >= img_width - 1 - BLUR_RADIUS) {
                        accessor_dst.write(coords, black);
                        return;
                    }

                    // Let's add vertical black border
                    if (item[1] <= BLUR_RADIUS ||
                        item[1] >= img_height - 1 - BLUR_RADIUS) {
                        accessor_dst.write(coords, black);
                        return;
                    }

                    cl::sycl::float4 tmp = (cl::sycl::float4)(0.f);
                    cl::sycl::uint4 rgba;

                    for (int i = item[0] - BLUR_RADIUS;
                         i < item[0] + BLUR_RADIUS;
                         i++) {
                        for (int j = item[1] - BLUR_RADIUS;
                             j < item[1] + BLUR_RADIUS;
                             j++) {
                            rgba = accessor_src.read(cl::sycl::int2(i, j));
                            // Sum over the square mask
                            tmp[0] += rgba.x();
                            tmp[1] += rgba.y();
                            tmp[2] += rgba.z();
                            // Keep alpha channel from anchor pixel
                            if (i == item[0] && j == item[1])
                                tmp[3] = rgba.w();
                        }
                    }
                    // Compute average intensity
                    tmp[0] /= BLUR_SIZE * BLUR_SIZE;
                    tmp[1] /= BLUR_SIZE * BLUR_SIZE;
                    tmp[2] /= BLUR_SIZE * BLUR_SIZE;

                    // Convert and write blur pixel
                    cl::sycl::uint4 tmp_u;
                    tmp_u[0] = tmp[0];
                    tmp_u[1] = tmp[1];
                    tmp_u[2] = tmp[2];
                    tmp_u[3] = tmp[3];

                    accessor_dst.write(coords, tmp_u);
                });
        });

        // Since we are in blocking execution mode for this sample simplicity,
        // we need to wait for the execution completeness.
        q.wait_and_throw();
    }
    catch (std::exception e) {
        std::cout << "SYCL exception caught: " << e.what() << "\n";
        return;
    }
    return;
}
#endif // BUILD_DPCPP
