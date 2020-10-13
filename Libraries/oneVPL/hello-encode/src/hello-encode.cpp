//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) encode application,
/// using oneVPL internal memory management
///
/// @file

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vpl/mfxdispatcher.h"
#include "vpl/mfxvideo.h"

#define MAX_PATH              260
#define MAX_WIDTH             3840
#define MAX_HEIGHT            2160
#define TARGETKBPS            4000
#define FRAMERATE             30
#define OUTPUT_FILE           "out.h265"
#define WAIT_100_MILLSECONDS  100
#define BITSTREAM_BUFFER_SIZE 2000000

#define VERIFY(x, y)       \
    if (!(x)) {            \
        printf("%s\n", y); \
        goto end;          \
    }

#define ALIGN16(value) (((value + 15) >> 4) << 4)

mfxStatus LoadRawFrame(mfxFrameSurface1 *surface, FILE *f);
void WriteEncodedStream(mfxBitstream &bs, FILE *f);
char *ValidateFileName(char *in);
mfxU16 ValidateSize(char *in, mfxU16 max);

void Usage(void) {
    printf("\n");
    printf("   Usage  :  hello-encode InputI420File width height\n\n");
    printf("             InputI420File    ... input file name (I420 raw frames)\n");
    printf("             width            ... input width\n");
    printf("             height           ... input height\n\n");
    printf("   Example:  hello-encode in.i420 128 96\n");
    printf("   To view:  ffplay %s\n\n", OUTPUT_FILE);
    printf(" * Encode I420 raw frames to HEVC/H265 elementary stream in %s\n\n", OUTPUT_FILE);
    return;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        Usage();
        return 1;
    }

    char *in_filename                = NULL;
    FILE *source                     = NULL;
    FILE *sink                       = NULL;
    mfxStatus sts                    = MFX_ERR_NONE;
    mfxLoader loader                 = NULL;
    mfxConfig cfg                    = NULL;
    mfxVariant impl_value            = { 0 };
    mfxSession session               = NULL;
    mfxU16 input_width               = 0;
    mfxU16 input_height              = 0;
    mfxBitstream bitstream           = { 0 };
    mfxVideoParam encode_params      = { 0 };
    mfxFrameSurface1 *enc_surface_in = NULL;
    mfxSyncPoint syncp               = { 0 };
    mfxU32 framenum                  = 0;
    bool is_draining                 = false;
    bool is_stillgoing               = true;

    // Setup input and output files
    in_filename = ValidateFileName(argv[1]);
    VERIFY(in_filename, "Input filename is not valid");

    source = fopen(in_filename, "rb");
    VERIFY(source, "Could not open input file");

    sink = fopen(OUTPUT_FILE, "wb");
    VERIFY(sink, "Could not create output file");

    input_width = ValidateSize(argv[2], MAX_WIDTH);
    VERIFY(input_width, "Input width is not valid");

    input_height = ValidateSize(argv[3], MAX_HEIGHT);
    VERIFY(input_height, "Input height is not valid");

    // Initialize VPL session for any implementation of HEVC/H265 encode
    loader = MFXLoad();
    VERIFY(NULL != loader, "MFXLoad failed");

    cfg = MFXCreateConfig(loader);
    VERIFY(NULL != cfg, "MFXCreateConfig failed")

    impl_value.Type     = MFX_VARIANT_TYPE_U32;
    impl_value.Data.U32 = MFX_CODEC_HEVC;
    sts                 = MFXSetConfigFilterProperty(
        cfg,
        (mfxU8 *)"mfxImplDescription.mfxEncoderDescription.encoder.CodecID",
        impl_value);
    VERIFY(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed");

    sts = MFXCreateSession(loader, 0, &session);
    VERIFY(MFX_ERR_NONE == sts, "Not able to create VPL session supporting HEVC/H265 encode");

    // Initialize encode parameters
    encode_params.mfx.CodecId                 = MFX_CODEC_HEVC;
    encode_params.mfx.TargetUsage             = MFX_TARGETUSAGE_BALANCED;
    encode_params.mfx.TargetKbps              = TARGETKBPS;
    encode_params.mfx.RateControlMethod       = MFX_RATECONTROL_VBR;
    encode_params.mfx.FrameInfo.FrameRateExtN = FRAMERATE;
    encode_params.mfx.FrameInfo.FrameRateExtD = 1;
    encode_params.mfx.FrameInfo.FourCC        = MFX_FOURCC_I420;
    encode_params.mfx.FrameInfo.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    encode_params.mfx.FrameInfo.CropW         = input_width;
    encode_params.mfx.FrameInfo.CropH         = input_height;
    encode_params.mfx.FrameInfo.Width         = ALIGN16(input_width);
    encode_params.mfx.FrameInfo.Height        = ALIGN16(input_height);

    encode_params.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;

    // Initialize the encoder
    sts = MFXVideoENCODE_Init(session, &encode_params);
    VERIFY(MFX_ERR_NONE == sts, "Encode init failed");

    // Prepare output bitstream and start encoding
    bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
    bitstream.Data      = (mfxU8 *)malloc(bitstream.MaxLength * sizeof(mfxU8));

    printf("Encoding %s -> %s\n", in_filename, OUTPUT_FILE);

    while (is_stillgoing == true) {
        // Load a new frame if not draining
        if (is_draining == false) {
            sts = MFXMemory_GetSurfaceForEncode(session, &enc_surface_in);
            VERIFY(MFX_ERR_NONE == sts, "Could not get encode surface");

            // Map makes surface writable by CPU for all implementations
            sts = enc_surface_in->FrameInterface->Map(enc_surface_in, MFX_MAP_WRITE);
            VERIFY(MFX_ERR_NONE == sts, "mfxFrameSurfaceInterface->Map failed");

            sts = LoadRawFrame(enc_surface_in, source);
            if (sts != MFX_ERR_NONE)
                is_draining = true;

            // Unmap/release returns local device access for all implementations
            sts = enc_surface_in->FrameInterface->Unmap(enc_surface_in);
            VERIFY(MFX_ERR_NONE == sts, "mfxFrameSurfaceInterface->Unmap failed");

            enc_surface_in->FrameInterface->Release(enc_surface_in);
            VERIFY(MFX_ERR_NONE == sts, "mfxFrameSurfaceInterface->Release failed");
        }

        sts = MFXVideoENCODE_EncodeFrameAsync(session,
                                              NULL,
                                              (is_draining == true) ? NULL : enc_surface_in,
                                              &bitstream,
                                              &syncp);

        switch (sts) {
            case MFX_ERR_NONE:
                // MFX_ERR_NONE and syncp indicate output is available
                if (syncp) {
                    // Encode output is not available on CPU until sync operation completes
                    sts = MFXVideoCORE_SyncOperation(session, syncp, WAIT_100_MILLSECONDS);
                    VERIFY(MFX_ERR_NONE == sts, "MFXVideoCORE_SyncOperation error");

                    WriteEncodedStream(bitstream, sink);
                    framenum++;
                }
                break;
            case MFX_ERR_NOT_ENOUGH_BUFFER:
                // This example deliberatly uses a large output buffer with immediate write to disk
                // for simplicity.
                // Handle when frame size exceeds available buffer here
                break;
            case MFX_ERR_MORE_DATA:
                // The function requires more data to generate any output
                if (is_draining == true)
                    is_stillgoing = false;
                break;
            case MFX_ERR_DEVICE_LOST:
                // For non-CPU implementations,
                // Cleanup if device is lost
                break;
            case MFX_WRN_DEVICE_BUSY:
                // For non-CPU implementations,
                // Wait a few milliseconds then try again
                break;
            case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:
                // The CPU reference implementation does not include mfxEncodeCtrl, but for
                // other implementations issues with mfxEncodeCtrl parameters can be handled here
                break;
            default:
                printf("unknown status %d\n", sts);
                is_stillgoing = false;
                break;
        }
    }

end:
    printf("Encoded %d frames\n", framenum);

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

// Write encoded stream to file
void WriteEncodedStream(mfxBitstream &bs, FILE *f) {
    fwrite(bs.Data + bs.DataOffset, 1, bs.DataLength, f);
    bs.DataLength = 0;
    return;
}

// Load raw I420 frame to mfxFrameSurface
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
            printf("Unsupported FourCC code, skip LoadRawFrame\n");
            break;
    }

    return MFX_ERR_NONE;
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