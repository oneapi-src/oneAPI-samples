//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) VPP application,
/// using oneVPL internal memory management for VPP in and
/// external memory for VPP out
///
/// @file

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vpl/mfxdispatcher.h"
#include "vpl/mfxvideo.h"

#define MAX_PATH             260
#define MAX_WIDTH            3840
#define MAX_HEIGHT           2160
#define OUTPUT_WIDTH         640
#define OUTPUT_HEIGHT        480
#define FRAMERATE            30
#define OUTPUT_FILE          "out.i420"
#define WAIT_100_MILLSECONDS 100

#define VERIFY(x, y)       \
    if (!(x)) {            \
        printf("%s\n", y); \
        goto end;          \
    }

#define ALIGN16(value) (((value + 15) >> 4) << 4)

mfxStatus LoadRawFrame(mfxFrameSurface1 *surface, FILE *f);
void WriteRawFrame(mfxFrameSurface1 *surface, FILE *f);
char *ValidateFileName(char *in);
mfxU16 ValidateSize(char *in, mfxU16 max);
mfxU32 GetSurfaceSize(mfxU32 fourcc, mfxU16 width, mfxU16 height);

void Usage(void) {
    printf("\n");
    printf("   Usage  :  hello-vpp InputI420File width height\n\n");
    printf("             InputI420File    ... input file name (i420 raw frames)\n");
    printf("             width            ... input width\n");
    printf("             height           ... input height\n\n");
    printf("   Example:  hello-vpp in.i420 128 96\n");
    printf(
        "   To view:  ffplay -f rawvideo -pixel_format yuv420p -video_size %dx%d -pixel_format yuv420p %s\n\n",
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT,
        OUTPUT_FILE);
    printf(" * Resize I420 raw frames to %dx%d size in %s\n\n",
           OUTPUT_WIDTH,
           OUTPUT_HEIGHT,
           OUTPUT_FILE);
    return;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        Usage();
        return 1;
    }

    char *in_filename                   = NULL;
    FILE *source                        = NULL;
    FILE *sink                          = NULL;
    mfxStatus sts                       = MFX_ERR_NONE;
    mfxLoader loader                    = NULL;
    mfxConfig cfg                       = NULL;
    mfxVariant impl_value               = { 0 };
    mfxSession session                  = NULL;
    mfxU16 input_width                  = 0;
    mfxU16 input_height                 = 0;
    mfxU16 out_width                    = 0;
    mfxU16 out_height                   = 0;
    mfxVideoParam vpp_params            = {};
    mfxFrameAllocRequest vpp_request[2] = { 0 };
    mfxU16 num_surfaces_in              = 0;
    mfxU16 num_surfaces_out             = 0;
    mfxU32 surface_size                 = 0;
    mfxU8 *vpp_data_out                 = NULL;
    mfxFrameSurface1 *vpp_surfaces_in   = NULL;
    mfxFrameSurface1 *vpp_surfaces_out  = NULL;
    int available_surface_index         = 0;
    mfxSyncPoint syncp                  = { 0 };
    mfxU32 framenum                     = 0;
    bool is_draining                    = false;
    bool is_stillgoing                  = true;
    mfxU16 i;

    // Setup input and output files
    in_filename = ValidateFileName(argv[1]);
    VERIFY(in_filename, "Input filename is not valid");

    source = fopen(in_filename, "rb");
    VERIFY(source, "Could not open input file");

    sink = fopen(OUTPUT_FILE, "wbx");
    VERIFY(sink, "Could not create output file");

    input_width = ValidateSize(argv[2], MAX_WIDTH);
    VERIFY(input_width, "Input width is not valid");

    input_height = ValidateSize(argv[3], MAX_HEIGHT);
    VERIFY(input_height, "Input height is not valid");

    // Initialize VPL session for video processing
    loader = MFXLoad();
    VERIFY(NULL != loader, "MFXLoad failed");

    cfg = MFXCreateConfig(loader);
    VERIFY(NULL != cfg, "MFXCreateConfig failed")

    impl_value.Type     = MFX_VARIANT_TYPE_U32;
    impl_value.Data.U32 = MFX_EXTBUFF_VPP_SCALING;
    sts                 = MFXSetConfigFilterProperty(
        cfg,
        (mfxU8 *)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
        impl_value);
    VERIFY(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed");

    sts = MFXCreateSession(loader, 0, &session);
    VERIFY(MFX_ERR_NONE == sts, "Not able to create VPL session supporting VPP");

    // Initialize VPP parameters
    // Input data
    vpp_params.vpp.In.FourCC        = MFX_FOURCC_I420;
    vpp_params.vpp.In.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    vpp_params.vpp.In.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    vpp_params.vpp.In.FrameRateExtN = FRAMERATE;
    vpp_params.vpp.In.FrameRateExtD = 1;
    vpp_params.vpp.In.CropW         = input_width;
    vpp_params.vpp.In.CropH         = input_height;
    vpp_params.vpp.In.Width         = ALIGN16(input_width);
    vpp_params.vpp.In.Height        = ALIGN16(input_height);
    // Output data - change output size to OUTPUT_WIDTH, OUTPUT_HEIGHT
    vpp_params.vpp.Out.FourCC        = MFX_FOURCC_I420;
    vpp_params.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    vpp_params.vpp.Out.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    vpp_params.vpp.Out.FrameRateExtN = FRAMERATE;
    vpp_params.vpp.Out.FrameRateExtD = 1;
    vpp_params.vpp.Out.CropW         = OUTPUT_WIDTH;
    vpp_params.vpp.Out.CropH         = OUTPUT_HEIGHT;
    vpp_params.vpp.Out.Width         = ALIGN16(OUTPUT_WIDTH);
    vpp_params.vpp.Out.Height        = ALIGN16(OUTPUT_HEIGHT);

    vpp_params.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

    // Query number of required surfaces for VPP
    sts = MFXVideoVPP_QueryIOSurf(session, &vpp_params, vpp_request);
    VERIFY(MFX_ERR_NONE == sts, "QueryIOSurf error");

    num_surfaces_in  = vpp_request[0].NumFrameSuggested;
    num_surfaces_out = vpp_request[1].NumFrameSuggested;

    // Allocate surfaces for VPP out
    // Frame surface array keeps pointers to all surface planes and general
    // frame info
    out_width  = vpp_params.vpp.Out.Width;
    out_height = vpp_params.vpp.Out.Height;

    surface_size = GetSurfaceSize(MFX_FOURCC_I420, out_width, out_height);
    VERIFY(surface_size, "VPP out surface size is wrong");

    vpp_data_out = (mfxU8 *)calloc(num_surfaces_out, surface_size);
    VERIFY(vpp_data_out, "Could not allocate buffer for VPP output frames");

    vpp_surfaces_out = (mfxFrameSurface1 *)calloc(num_surfaces_out, sizeof(mfxFrameSurface1));
    VERIFY(vpp_surfaces_out, "Could not allocate VPP output surfaces");

    for (i = 0; i < num_surfaces_out; i++) {
        vpp_surfaces_out[i].Info   = vpp_params.vpp.Out;
        vpp_surfaces_out[i].Data.Y = &vpp_data_out[surface_size * i];
        vpp_surfaces_out[i].Data.U = vpp_surfaces_out[i].Data.Y + out_width * out_height;
        vpp_surfaces_out[i].Data.V =
            vpp_surfaces_out[i].Data.U + ((out_width / 2) * (out_height / 2));
        vpp_surfaces_out[i].Data.Pitch = out_width;
    }

    // Initialize VPP and start processing
    sts = MFXVideoVPP_Init(session, &vpp_params);
    VERIFY(MFX_ERR_NONE == sts, "Could not initialize VPP");

    printf("Processing %s -> %s\n", in_filename, OUTPUT_FILE);

    while (is_stillgoing == true) {
        // Load a new frame if not draining
        if (is_draining == false) {
            vpp_surfaces_in = NULL;

            sts = MFXMemory_GetSurfaceForVPP(session, &vpp_surfaces_in);
            VERIFY(MFX_ERR_NONE == sts, "Unknown error in MFXMemory_GetSurfaceForVPP");

            // Map makes surface writable by CPU for all implementations
            sts = vpp_surfaces_in->FrameInterface->Map(vpp_surfaces_in, MFX_MAP_WRITE);
            VERIFY(MFX_ERR_NONE == sts, "mfxFrameSurfaceInterface->Map failed");

            sts = LoadRawFrame(vpp_surfaces_in, source);
            if (sts != MFX_ERR_NONE)
                is_draining = true;

            // Unmap/release returns local device access for all implementations
            sts = vpp_surfaces_in->FrameInterface->Unmap(vpp_surfaces_in);
            VERIFY(MFX_ERR_NONE == sts, "mfxFrameSurfaceInterface->Unmap failed");

            sts = vpp_surfaces_in->FrameInterface->Release(vpp_surfaces_in);
            VERIFY(MFX_ERR_NONE == sts, "mfxFrameSurfaceInterface->Release failed");
        }

        // Find free frame surface for VPP out
        available_surface_index = -1;
        for (int i = 0; i < num_surfaces_out; i++) {
            if (!vpp_surfaces_out[i].Data.Locked) {
                available_surface_index = i;
                break;
            }
        }
        VERIFY(available_surface_index >= 0, "Could not find available output surface");

        sts = MFXVideoVPP_RunFrameVPPAsync(session,
                                           (is_draining == true) ? NULL : vpp_surfaces_in,
                                           &vpp_surfaces_out[available_surface_index],
                                           NULL,
                                           &syncp);

        switch (sts) {
            case MFX_ERR_NONE:
                // MFX_ERR_NONE and syncp indicate output is available
                if (syncp) {
                    // VPP output is not available on CPU until sync operation completes
                    sts = MFXVideoCORE_SyncOperation(session, syncp, WAIT_100_MILLSECONDS);
                    VERIFY(MFX_ERR_NONE == sts, "MFXVideoCORE_SyncOperation error");

                    WriteRawFrame(&vpp_surfaces_out[available_surface_index], sink);
                    framenum++;
                }
                break;
            case MFX_ERR_MORE_DATA:
                // Need more input frames before VPP can produce an output
                if (is_draining == true)
                    is_stillgoing = false;
                break;
            case MFX_ERR_MORE_SURFACE:
                // The output frame is ready after synchronization.
                // Need more surfaces at output for additional output frames available.
                // This applies to external memory allocations and should not be expected for
                // a simple internal allocation case like this
                break;
            case MFX_ERR_DEVICE_LOST:
                // For non-CPU implementations,
                // Cleanup if device is lost
                break;
            case MFX_WRN_DEVICE_BUSY:
                // For non-CPU implementations,
                // Wait a few milliseconds then try again
                break;
            default:
                printf("unknown status %d\n", sts);
                is_stillgoing = false;
                break;
        }
    }

end:
    printf("Processed %d frames\n", framenum);

    // Clean up resources - It is recommended to close components first, before
    // releasing allocated surfaces, since some surfaces may still be locked by
    // internal resources.
    if (loader)
        MFXUnload(loader);

    if (vpp_surfaces_out) {
        free(vpp_surfaces_out);
    }

    if (vpp_data_out)
        free(vpp_data_out);

    if (source)
        fclose(source);

    if (sink)
        fclose(sink);

    return 0;
}

// Load raw I420 frames to mfxFrameSurface
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
            printf("Unsupported FourCC code, skip LoadRawFrame\n");
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

    switch (info->FourCC) {
        case MFX_FOURCC_I420:
            // write luminance plane (Y)
            pitch = data->Pitch;
            for (i = 0; i < h; i++) {
                fwrite(data->Y + i * pitch, 1, w, f);
            }

            // write chrominance (U, V)
            pitch /= 2;
            h /= 2;
            w /= 2;
            for (i = 0; i < h; i++) {
                fwrite(data->U + i * pitch, 1, w, f);
            }

            for (i = 0; i < h; i++) {
                fwrite(data->V + i * pitch, 1, w, f);
            }
            break;
        default:
            printf("Unsupported FourCC code, skip WriteRawFrame\n");
            break;
    }

    return;
}

// Return the surface size in bytes given format and dimensions
mfxU32 GetSurfaceSize(mfxU32 fourcc, mfxU16 width, mfxU16 height) {
    mfxU32 bytes = 0;

    switch (fourcc) {
        case MFX_FOURCC_I420:
            bytes = width * height + (width >> 1) * (height >> 1) + (width >> 1) * (height >> 1);
            break;
        default:
            break;
    }

    return bytes;
}

char *ValidateFileName(char *in) {
    if (in) {
        if (strlen(in) > MAX_PATH)
            return NULL;
    }

    return in;
}

mfxU16 ValidateSize(char *in, mfxU16 max) {
    mfxU16 isize = (mfxU16)strtol(in, NULL, 10);
    if (isize <= 0 || isize > max) {
        return 0;
    }

    return isize;
}
