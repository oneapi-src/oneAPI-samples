//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) vpp application.
///
/// @file

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vpl/mfxvideo.h"

#define MAX_PATH   260
#define MAX_WIDTH  3840
#define MAX_HEIGHT 2160

#define OUTPUT_FILE   "out.i420"
#define OUTPUT_WIDTH  640
#define OUTPUT_HEIGHT 480

mfxStatus LoadRawFrame(mfxFrameSurface1 *surface, FILE *f);
void WriteRawFrame(mfxFrameSurface1 *surface, FILE *f);
mfxU32 GetSurfaceSize(mfxU32 fourcc, mfxU32 width, mfxU32 height);
mfxI32 GetFreeSurfaceIndex(const std::vector<mfxFrameSurface1> &surface_pool);
char *ValidateFileName(char *in);

// Print usage message
void Usage(void) {
    printf("Usage: hello-vpp SOURCE WIDTH HEIGHT\n\n"
           "Process raw I420 video in SOURCE having dimensions WIDTH x HEIGHT "
           "to resized I420 raw video in %s\n\n"
           "To view:\n"
           " ffplay -video_size [width]x[height] "
           "-pixel_format yuv420p -f rawvideo %s\n",
           OUTPUT_FILE,
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

    // initialize session
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

    // For simplistic memory management, system memory surfaces are used to
    // store the raw frames (Note that when using HW acceleration video surfaces
    // are prefered, for better performance)
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
    // Output data
    vpp_params.vpp.Out.FourCC        = MFX_FOURCC_I420;
    vpp_params.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    vpp_params.vpp.Out.CropX         = 0;
    vpp_params.vpp.Out.CropY         = 0;
    vpp_params.vpp.Out.CropW         = OUTPUT_WIDTH;
    vpp_params.vpp.Out.CropH         = OUTPUT_HEIGHT;
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

    mfxU16 num_surfaces_in  = vpp_request[0].NumFrameSuggested;
    mfxU16 num_surfaces_out = vpp_request[1].NumFrameSuggested;

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

    std::vector<mfxU8> buffers_in(surface_size * num_surfaces_in);
    mfxU8 *buffers_in_data = buffers_in.data();

    std::vector<mfxFrameSurface1> vpp_surfaces_in(num_surfaces_in);
    for (mfxI32 i = 0; i < num_surfaces_in; i++) {
        memset(&vpp_surfaces_in[i], 0, sizeof(mfxFrameSurface1));
        vpp_surfaces_in[i].Info   = vpp_params.vpp.In;
        vpp_surfaces_in[i].Data.Y = &buffers_in_data[surface_size * i];
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

    std::vector<mfxU8> buffers_out(surface_size * num_surfaces_out);
    mfxU8 *buffers_out_data = buffers_out.data();

    std::vector<mfxFrameSurface1> vpp_surfaces_out(num_surfaces_out);
    for (mfxI32 i = 0; i < num_surfaces_out; i++) {
        memset(&vpp_surfaces_out[i], 0, sizeof(mfxFrameSurface1));
        vpp_surfaces_out[i].Info   = vpp_params.vpp.Out;
        vpp_surfaces_out[i].Data.Y = &buffers_out_data[surface_size * i];
        vpp_surfaces_out[i].Data.U =
            vpp_surfaces_out[i].Data.Y + width * height;
        vpp_surfaces_out[i].Data.V =
            vpp_surfaces_out[i].Data.U + ((width / 2) * (height / 2));
        vpp_surfaces_out[i].Data.Pitch = width;
    }

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
    int index_in = 0, index_out = 0;
    mfxSyncPoint syncp;
    mfxU32 framenum = 0;

    printf("Processing %s -> %s\n", in_filename, OUTPUT_FILE);

    // Stage 1: Main processing loop
    while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts) {
        index_in =
            GetFreeSurfaceIndex(vpp_surfaces_in); // Find free frame surface
        if (index_in == MFX_ERR_NOT_FOUND) {
            fclose(source);
            fclose(sink);
            puts("no available surface");
            return 1;
        }

        sts = LoadRawFrame(&vpp_surfaces_in[index_in], source);
        if (sts != MFX_ERR_NONE)
            break;

        index_out = GetFreeSurfaceIndex(
            vpp_surfaces_out); // Find free output frame surface
        if (index_out == MFX_ERR_NOT_FOUND) {
            fclose(source);
            fclose(sink);
            puts("no available surface");
            return 1;
        }

        for (;;) {
            // Process a frame asynchronously (returns immediately)
            sts = MFXVideoVPP_RunFrameVPPAsync(session,
                                               &vpp_surfaces_in[index_in],
                                               &vpp_surfaces_out[index_out],
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
            WriteRawFrame(&vpp_surfaces_out[index_out], sink);
            bitstream.DataLength = 0;
        }
    }

    sts = MFX_ERR_NONE;

    // Stage 2: Retrieve the buffered processed frames
    while (MFX_ERR_NONE <= sts) {
        index_out =
            GetFreeSurfaceIndex(vpp_surfaces_out); // Find free frame surface
        if (index_out == MFX_ERR_NOT_FOUND) {
            fclose(source);
            fclose(sink);
            puts("no available surface");
            return 1;
        }

        for (;;) {
            // Process a frame asynchronously (returns immediately)
            sts = MFXVideoVPP_RunFrameVPPAsync(session,
                                               NULL,
                                               &vpp_surfaces_out[index_out],
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
            WriteRawFrame(&vpp_surfaces_out[index_out], sink);
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
    mfxU32 bytes;
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
                bytes = (mfxU32)fread(ptr + i * pitch, 1, w, f);
                if (w != bytes)
                    return MFX_ERR_MORE_DATA;
            }

            // read chrominance (U, V)
            pitch /= 2;
            h /= 2;
            w /= 2;
            ptr = data->U;
            for (i = 0; i < h; i++) {
                bytes = (mfxU32)fread(ptr + i * pitch, 1, w, f);
                if (w != bytes)
                    return MFX_ERR_MORE_DATA;
            }

            ptr = data->V;
            for (i = 0; i < h; i++) {
                bytes = (mfxU32)fread(ptr + i * pitch, 1, w, f);
                if (w != bytes)
                    return MFX_ERR_MORE_DATA;
            }
            break;
        default:
            break;
    }

    return MFX_ERR_NONE;
}

// Write raw I420 frame to file
void WriteRawFrame(mfxFrameSurface1 *surface, FILE *f) {
    mfxU16 w, h, i, pitch;
    mfxFrameInfo *info = &surface->Info;
    mfxFrameData *data = &surface->Data;

    w = info->Width;
    h = info->Height;

    // write the output to disk
    switch (info->FourCC) {
        case MFX_FOURCC_I420:
            // Y
            pitch = data->Pitch;
            for (i = 0; i < h; i++) {
                fwrite(data->Y + i * pitch, 1, w, f);
            }
            // U
            pitch /= 2;
            h /= 2;
            w /= 2;
            for (i = 0; i < h; i++) {
                fwrite(data->U + i * pitch, 1, w, f);
            }
            // V
            for (i = 0; i < h; i++) {
                fwrite(data->V + i * pitch, 1, w, f);
            }
            break;
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
