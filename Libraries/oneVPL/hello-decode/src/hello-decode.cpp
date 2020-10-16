//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) decode application,
/// using oneVPL internal memory management
///
/// @file

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vpl/mfxdispatcher.h"
#include "vpl/mfxvideo.h"

#define MAX_PATH              260
#define OUTPUT_FILE           "out.i420"
#define WAIT_100_MILLSECONDS  100
#define BITSTREAM_BUFFER_SIZE 2000000

#define VERIFY(x, y)       \
    if (!(x)) {            \
        printf("%s\n", y); \
        goto end;          \
    }

#define ALIGN16(value) (((value + 15) >> 4) << 4)

mfxStatus ReadEncodedStream(mfxBitstream &bs, FILE *f);
void WriteRawFrame(mfxFrameSurface1 *surface, FILE *f);
char *ValidateFileName(char *in);

void Usage(void) {
    printf("\n");
    printf("   Usage  :  hello-decode InputBitstream\n\n");
    printf("             InputBitstream   ... input file name (HEVC/H265 elementary stream)\n\n");
    printf("   Example:  hello-decode in.h265\n");
    printf(
        "   To view:  ffplay -f rawvideo -pixel_format yuv420p -video_size [width]x[height] %s\n\n",
        OUTPUT_FILE);
    printf(" * Decode HEVC/H265 elementary stream to I420 raw frames in %s\n\n", OUTPUT_FILE);
    return;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        Usage();
        return 1;
    }

    char *in_filename                 = NULL;
    FILE *source                      = NULL;
    FILE *sink                        = NULL;
    mfxStatus sts                     = MFX_ERR_NONE;
    mfxLoader loader                  = NULL;
    mfxConfig cfg                     = NULL;
    mfxVariant impl_value             = { 0 };
    mfxSession session                = NULL;
    mfxBitstream bitstream            = { 0 };
    mfxFrameSurface1 *dec_surface_out = NULL;
    mfxSyncPoint syncp                = { 0 };
    mfxU32 framenum                   = 0;
    bool is_draining                  = false;
    bool is_stillgoing                = true;

    // Setup input and output files
    in_filename = ValidateFileName(argv[1]);
    VERIFY(in_filename, "Input filename is not valid");

    source = fopen(in_filename, "rb");
    VERIFY(source, "Could not open input file");

    sink = fopen(OUTPUT_FILE, "wb");
    VERIFY(sink, "Could not create output file");

    // Initialize VPL session for any implementation of HEVC/H265 decode
    loader = MFXLoad();
    VERIFY(NULL != loader, "MFXLoad failed");

    cfg = MFXCreateConfig(loader);
    VERIFY(NULL != cfg, "MFXCreateConfig failed")

    impl_value.Type     = MFX_VARIANT_TYPE_U32;
    impl_value.Data.U32 = MFX_CODEC_HEVC;
    sts                 = MFXSetConfigFilterProperty(
        cfg,
        (mfxU8 *)"mfxImplDescription.mfxDecoderDescription.decoder.CodecID",
        impl_value);
    VERIFY(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed");

    sts = MFXCreateSession(loader, 0, &session);
    VERIFY(MFX_ERR_NONE == sts, "Not able to create VPL session supporting HEVC/H265 decode");

    // Prepare input bitstream and start decoding
    bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
    bitstream.Data      = (mfxU8 *)malloc(bitstream.MaxLength * sizeof(mfxU8));
    VERIFY(bitstream.Data, "Not able to allocate input buffer");
    memset(bitstream.Data, 0, bitstream.MaxLength * sizeof(mfxU8));
    bitstream.CodecId = MFX_CODEC_HEVC;

    printf("Decoding %s -> %s\n", in_filename, OUTPUT_FILE);

    while (is_stillgoing == true) {
        // Load encoded stream if not draining
        if (is_draining == false) {
            sts = ReadEncodedStream(bitstream, source);
            if (sts != MFX_ERR_NONE)
                is_draining = true;
        }

        sts = MFXVideoDECODE_DecodeFrameAsync(session,
                                              (is_draining) ? NULL : &bitstream,
                                              NULL,
                                              &dec_surface_out,
                                              &syncp);

        switch (sts) {
            case MFX_ERR_NONE:
                do {
                    sts = dec_surface_out->FrameInterface->Synchronize(dec_surface_out,
                                                                       WAIT_100_MILLSECONDS);
                    if (MFX_ERR_NONE == sts) {
                        sts = dec_surface_out->FrameInterface->Map(dec_surface_out, MFX_MAP_READ);
                        VERIFY(MFX_ERR_NONE == sts, "mfxFrameSurfaceInterface->Map failed");

                        WriteRawFrame(dec_surface_out, sink);
                        framenum++;

                        sts = dec_surface_out->FrameInterface->Unmap(dec_surface_out);
                        VERIFY(MFX_ERR_NONE == sts, "mfxFrameSurfaceInterface->Unmap failed");

                        sts = dec_surface_out->FrameInterface->Release(dec_surface_out);
                        VERIFY(MFX_ERR_NONE == sts, "mfxFrameSurfaceInterface->Release failed");
                    }
                } while (sts == MFX_WRN_IN_EXECUTION);
                break;
            case MFX_ERR_MORE_DATA:
                // The function requires more bitstream at input before decoding can proceed
                if (is_draining)
                    is_stillgoing = false;
                break;
            case MFX_ERR_MORE_SURFACE:
                // The function requires more frame surface at output before decoding can proceed.
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
            case MFX_WRN_VIDEO_PARAM_CHANGED:
                // The decoder detected a new sequence header in the bitstream.
                // Video parameters may have changed.
                // In external memory allocation case, might need to reallocate the output surface
                break;
            case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:
                // The function detected that video parameters provided by the application
                // are incompatible with initialization parameters.
                // The application should close the component and then reinitialize it
                break;
            case MFX_ERR_REALLOC_SURFACE:
                // Bigger surface_work required. May be returned only if
                // mfxInfoMFX::EnableReallocRequest was set to ON during initialization.
                // This applies to external memory allocations and should not be expected for
                // a simple internal allocation case like this
                break;
            default:
                printf("unknown status %d\n", sts);
                is_stillgoing = false;
                break;
        }
    }

end:
    printf("Decoded %d frames\n", framenum);

    // Clean up resources - It is recommended to close components first, before
    // releasing allocated surfaces, since some surfaces may still be locked by
    // internal resources.
    if (loader)
        MFXUnload(loader);

    if (bitstream.Data)
        free(bitstream.Data);

    if (source)
        fclose(source);

    if (sink)
        fclose(sink);

    return 0;
}

// Read encoded stream from file
mfxStatus ReadEncodedStream(mfxBitstream &bs, FILE *f) {
    memmove(bs.Data, bs.Data + bs.DataOffset, bs.DataLength);
    bs.DataOffset = 0;
    bs.DataLength += (mfxU32)fread(bs.Data + bs.DataLength, 1, bs.MaxLength - bs.DataLength, f);
    if (bs.DataLength == 0)
        return MFX_ERR_MORE_DATA;

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
            printf("Unsupported FourCC code, skip WriteRawFrame\n");
            break;
    }

    return;
}

char *ValidateFileName(char *in) {
    if (in) {
        if (strlen(in) > MAX_PATH)
            return NULL;
    }

    return in;
}