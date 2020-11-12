//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) encode application.
///
/// @file

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vpl/mfxvideo.h"

#define MAX_PATH    260
#define OUTPUT_FILE "out.h265"
#define MAX_WIDTH   3840
#define MAX_HEIGHT  2160

mfxStatus LoadRawFrame(mfxFrameSurface1 *surface, FILE *f);
void WriteEncodedStream(mfxU8 *data, mfxU32 length, FILE *f);
mfxU32 GetSurfaceSize(mfxU32 fourcc, mfxU32 width, mfxU32 height);
mfxI32 GetFreeSurfaceIndex(const std::vector<mfxFrameSurface1> &surface_pool);
char *ValidateFileName(char *in);

// Print usage message
void Usage(void) {
    printf("Usage: hello-encode SOURCE WIDTH HEIGHT\n\n"
           "Encode raw I420 video in SOURCE having dimensions WIDTH x HEIGHT "
           "to H265 in %s\n\n"
           "To view:\n"
           " ffplay %s\n",
           OUTPUT_FILE,
           OUTPUT_FILE);
}

int main(int argc, char *argv[]) {
    mfxU32 codec_id = MFX_CODEC_HEVC;
    mfxU32 fourcc   = MFX_FOURCC_I420;

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
        printf("could not open input file, \"%s\"\n", in_filename);
        return 1;
    }
    FILE *sink = fopen(OUTPUT_FILE, "wb");
    if (!sink) {
        fclose(source);
        printf("could not open output file, %s\n", OUTPUT_FILE);
        return 1;
    }
    mfxI32 isize = strtol(argv[2], NULL, 10);
    if (isize <= 0 || isize > MAX_WIDTH) {
        fclose(source);
        fclose(sink);
        puts("input size is not valid\n");
        return 1;
    }
    mfxI32 input_width = isize;

    isize = strtol(argv[3], NULL, 10);
    if (isize <= 0 || isize > MAX_HEIGHT) {
        fclose(source);
        fclose(sink);
        puts("input size is not valid\n");
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

    // Initialize encoder parameters
    mfxVideoParam encode_params;
    memset(&encode_params, 0, sizeof(encode_params));
    encode_params.mfx.CodecId                 = codec_id;
    encode_params.mfx.TargetUsage             = MFX_TARGETUSAGE_BALANCED;
    encode_params.mfx.TargetKbps              = 4000;
    encode_params.mfx.RateControlMethod       = MFX_RATECONTROL_VBR;
    encode_params.mfx.FrameInfo.FrameRateExtN = 30;
    encode_params.mfx.FrameInfo.FrameRateExtD = 1;
    encode_params.mfx.FrameInfo.FourCC        = fourcc;
    encode_params.mfx.FrameInfo.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    encode_params.mfx.FrameInfo.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    encode_params.mfx.FrameInfo.CropX         = 0;
    encode_params.mfx.FrameInfo.CropY         = 0;
    encode_params.mfx.FrameInfo.CropW         = input_width;
    encode_params.mfx.FrameInfo.CropH         = input_height;
    // Width must be a multiple of 16
    encode_params.mfx.FrameInfo.Width =
        (((input_width + 15) >> 4) << 4); // 16 bytes alignment
    // Height must be a multiple of 16 in case of frame picture and a multiple
    // of 32 in case of field picture
    encode_params.mfx.FrameInfo.Height = (((input_height + 15) >> 4) << 4);

    encode_params.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;

    // Query number required surfaces for encoder
    mfxFrameAllocRequest encode_request = { 0 };
    sts = MFXVideoENCODE_QueryIOSurf(session, &encode_params, &encode_request);

    if (sts != MFX_ERR_NONE) {
        fclose(source);
        fclose(sink);
        puts("QueryIOSurf error");
        return 1;
    }

    // Determine the required number of surfaces for encoder
    mfxU16 num_encode_surfaces = encode_request.NumFrameSuggested;

    // Allocate surfaces for encoder

    // Frame surface array keeps pointers to all surface planes and general
    // frame info
    mfxU32 surface_size = GetSurfaceSize(fourcc, input_width, input_height);
    if (surface_size == 0) {
        fclose(source);
        fclose(sink);
        puts("Surface size is wrong");
        return 1;
    }

    std::vector<mfxU8> surface_buffers(surface_size * num_encode_surfaces);
    mfxU8 *surface_buffers_data = surface_buffers.data();

    // Allocate surface headers (mfxFrameSurface1) for encoder
    std::vector<mfxFrameSurface1> encode_surfaces(num_encode_surfaces);
    for (mfxI32 i = 0; i < num_encode_surfaces; i++) {
        memset(&encode_surfaces[i], 0, sizeof(mfxFrameSurface1));
        encode_surfaces[i].Info   = encode_params.mfx.FrameInfo;
        encode_surfaces[i].Data.Y = &surface_buffers_data[surface_size * i];

        encode_surfaces[i].Data.U =
            encode_surfaces[i].Data.Y + input_width * input_height;
        encode_surfaces[i].Data.V = encode_surfaces[i].Data.U +
                                    ((input_width / 2) * (input_height / 2));
        encode_surfaces[i].Data.Pitch = input_width;
    }

    // Initialize the encoder
    sts = MFXVideoENCODE_Init(session, &encode_params);
    if (sts != MFX_ERR_NONE) {
        fclose(source);
        fclose(sink);
        puts("could not initialize encode");
        return 1;
    }

    // Prepare bit stream buffer
    mfxBitstream bitstream = { 0 };
    bitstream.MaxLength    = 2000000;
    std::vector<mfxU8> bitstream_data(bitstream.MaxLength);
    bitstream.Data = bitstream_data.data();

    // Start encoding the frames
    mfxI32 index = 0;
    mfxSyncPoint syncp;
    mfxU32 framenum = 0;

    printf("Encoding %s -> %s\n", in_filename, OUTPUT_FILE);

    // Stage 1: Main encoding loop
    while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts) {
        index = GetFreeSurfaceIndex(encode_surfaces); // Find free frame surface
        if (index == MFX_ERR_NOT_FOUND) {
            fclose(source);
            fclose(sink);
            puts("no available surface");
            return 1;
        }

        sts = LoadRawFrame(&encode_surfaces[index], source);
        if (sts != MFX_ERR_NONE)
            break;

        for (;;) {
            // Encode a frame asynchronously (returns immediately)
            sts = MFXVideoENCODE_EncodeFrameAsync(session,
                                                  NULL,
                                                  &encode_surfaces[index],
                                                  &bitstream,
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
                60000); // Synchronize. Wait until encoded frame is ready
            ++framenum;
            WriteEncodedStream(bitstream.Data + bitstream.DataOffset,
                               bitstream.DataLength,

                               sink);
            bitstream.DataLength = 0;
        }
    }

    sts = MFX_ERR_NONE;

    // Stage 2: Retrieve the buffered encoded frames
    while (MFX_ERR_NONE <= sts) {
        for (;;) {
            // Encode a frame asynchronously (returns immediately)
            sts = MFXVideoENCODE_EncodeFrameAsync(session,
                                                  NULL,
                                                  NULL,
                                                  &bitstream,
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
                60000); // Synchronize. Wait until encoded frame is ready

            ++framenum;
            WriteEncodedStream(bitstream.Data + bitstream.DataOffset,
                               bitstream.DataLength,

                               sink);

            bitstream.DataLength = 0;
        }
    }

    printf("Encoded %d frames\n", framenum);

    // Clean up resources - It is recommended to close components first, before
    // releasing allocated surfaces, since some surfaces may still be locked by
    // internal resources.
    MFXVideoENCODE_Close(session);

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

// Write encoded stream to file
void WriteEncodedStream(mfxU8 *data, mfxU32 length, FILE *f) {
    fwrite(data, 1, length, f);
}

// Return the surface size in bytes given format and dimensions
mfxU32 GetSurfaceSize(mfxU32 fourcc, mfxU32 width, mfxU32 height) {
    mfxU32 nbytes = 0;

    switch (fourcc) {
        case MFX_FOURCC_I420:
            nbytes = width * height + (width >> 1) * (height >> 1) +
                     (width >> 1) * (height >> 1);
            break;
        default:
            break;
    }

    return nbytes;
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
