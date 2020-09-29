//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) decode application.
///
/// @file

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vpl/mfxvideo.h"

#define MAX_PATH    260
#define OUTPUT_FILE "out.i420"

mfxStatus ReadEncodedStream(mfxBitstream &bs, mfxU32 codecid, FILE *f);
void WriteRawFrame(mfxFrameSurface1 *surface, FILE *f);
mfxU32 GetSurfaceSize(mfxU32 fourcc, mfxU32 width, mfxU32 height);
int GetFreeSurfaceIndex(mfxFrameSurface1 *surface_pool, mfxU16 pool_size);
char *ValidateFileName(char *in);

// Print usage message
void Usage(void) {
    printf("Usage: hello-decode SOURCE\n\n"
           "Decode H265/HEVC video in SOURCE "
           "to I420 raw video in %s\n\n"
           "To view:\n"
           " ffplay -video_size [width]x[height] "
           "-pixel_format yuv420p -f rawvideo %s\n",
           OUTPUT_FILE,
           OUTPUT_FILE);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
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
        printf("could not open input file, %s\n", in_filename);
        return 1;
    }
    FILE *sink = fopen(OUTPUT_FILE, "wb");
    if (!sink) {
        fclose(source);
        printf("could not create output file, \"%s\"\n", OUTPUT_FILE);
        return 1;
    }

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

    // prepare input bitstream
    mfxBitstream bitstream = { 0 };
    bitstream.MaxLength    = 2000000;
    std::vector<mfxU8> input_buffer;
    input_buffer.resize(bitstream.MaxLength);
    bitstream.Data = input_buffer.data();

    mfxU32 codec_id = MFX_CODEC_HEVC;
    ReadEncodedStream(bitstream, codec_id, source);

    // initialize decode parameters from stream header
    mfxVideoParam decode_params;
    memset(&decode_params, 0, sizeof(decode_params));
    decode_params.mfx.CodecId = codec_id;
    decode_params.IOPattern   = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    sts = MFXVideoDECODE_DecodeHeader(session, &bitstream, &decode_params);
    if (sts != MFX_ERR_NONE) {
        fclose(source);
        fclose(sink);
        printf("Problem decoding header.  DecodeHeader sts=%d\n", sts);
        return 1;
    }

    // Query number required surfaces for decoder
    mfxFrameAllocRequest decode_request = { 0 };
    MFXVideoDECODE_QueryIOSurf(session, &decode_params, &decode_request);

    // Determine the required number of surfaces for decoder output
    mfxU16 num_decode_surfaces = decode_request.NumFrameSuggested;

    std::vector<mfxFrameSurface1> decode_surfaces;
    decode_surfaces.resize(num_decode_surfaces);
    mfxFrameSurface1 *decode_surfaces_data = decode_surfaces.data();
    if (decode_surfaces_data == NULL) {
        fclose(source);
        fclose(sink);
        puts("Fail to allocate decode frame memory surface");
        return 1;
    }
    // initialize surface pool for decode (I420 format)
    mfxU32 surface_size = GetSurfaceSize(decode_params.mfx.FrameInfo.FourCC,
                                         decode_params.mfx.FrameInfo.Width,
                                         decode_params.mfx.FrameInfo.Height);
    if (surface_size == 0) {
        fclose(source);
        fclose(sink);
        puts("Surface size is wrong");
        return 1;
    }
    size_t frame_pool_buffer_size =
        static_cast<size_t>(surface_size) * num_decode_surfaces;
    std::vector<mfxU8> output_buffer;
    output_buffer.resize(frame_pool_buffer_size);
    mfxU8 *output_buffer_data = output_buffer.data();

    mfxU16 surface_w = (decode_params.mfx.FrameInfo.FourCC == MFX_FOURCC_I010)
                           ? decode_params.mfx.FrameInfo.Width * 2
                           : decode_params.mfx.FrameInfo.Width;
    mfxU16 surface_h = decode_params.mfx.FrameInfo.Height;

    for (mfxU32 i = 0; i < num_decode_surfaces; i++) {
        decode_surfaces_data[i]        = { 0 };
        decode_surfaces_data[i].Info   = decode_params.mfx.FrameInfo;
        size_t buf_offset              = static_cast<size_t>(i) * surface_size;
        decode_surfaces_data[i].Data.Y = output_buffer_data + buf_offset;
        decode_surfaces_data[i].Data.U =
            output_buffer_data + buf_offset + (surface_w * surface_h);
        decode_surfaces_data[i].Data.V = decode_surfaces_data[i].Data.U +
                                         ((surface_w / 2) * (surface_h / 2));
        decode_surfaces_data[i].Data.Pitch = surface_w;
    }

    // input parameters finished, now initialize decode
    sts = MFXVideoDECODE_Init(session, &decode_params);
    if (sts != MFX_ERR_NONE) {
        fclose(source);
        fclose(sink);
        puts("Could not initialize decode");
        exit(1);
    }
    // ------------------
    // main loop
    // ------------------
    int framenum                     = 0;
    mfxSyncPoint syncp               = { 0 };
    mfxFrameSurface1 *output_surface = nullptr;

    printf("Decoding %s -> %s\n", in_filename, OUTPUT_FILE);
    for (;;) {
        bool stillgoing = true;
        int index =
            GetFreeSurfaceIndex(decode_surfaces_data, num_decode_surfaces);
        while (stillgoing) {
            // submit async decode request
            sts = MFXVideoDECODE_DecodeFrameAsync(session,
                                                  &bitstream,
                                                  &decode_surfaces_data[index],
                                                  &output_surface,
                                                  &syncp);

            // next step actions provided by application
            switch (sts) {
                case MFX_ERR_MORE_DATA: // more data is needed to decode
                    ReadEncodedStream(bitstream, codec_id, source);
                    if (bitstream.DataLength == 0)
                        stillgoing = false; // stop if end of file
                    break;
                case MFX_ERR_MORE_SURFACE: // feed a fresh surface to decode
                    index = GetFreeSurfaceIndex(decode_surfaces_data,
                                                num_decode_surfaces);
                    break;
                case MFX_ERR_NONE: // no more steps needed, exit loop
                    stillgoing = false;
                    break;
                default: // state is not one of the cases above
                    printf("Error in DecodeFrameAsync: sts=%d\n", sts);
                    exit(1);
                    break;
            }
        }

        if (sts < 0)
            break;

        // data available to app only after sync
        MFXVideoCORE_SyncOperation(session, syncp, 60000);

        // write output if output file specified
        if (sink)
            WriteRawFrame(output_surface, sink);

        framenum++;
    }

    sts = MFX_ERR_NONE;
    memset(&syncp, 0, sizeof(mfxSyncPoint));
    output_surface = nullptr;

    // retrieve the buffered decoded frames
    while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_SURFACE == sts) {
        int index =
            GetFreeSurfaceIndex(decode_surfaces_data,
                                num_decode_surfaces); // Find free frame surface

        // Decode a frame asynchronously (returns immediately)

        sts = MFXVideoDECODE_DecodeFrameAsync(session,
                                              NULL,
                                              &decode_surfaces_data[index],
                                              &output_surface,
                                              &syncp);

        // Ignore warnings if output is available, if no output and no action
        // required just repeat the DecodeFrameAsync call
        if (MFX_ERR_NONE < sts && syncp)
            sts = MFX_ERR_NONE;

        if (sts == MFX_ERR_NONE) {
            sts = MFXVideoCORE_SyncOperation(
                session,
                syncp,
                60000); // Synchronize. Waits until decoded frame is ready

            // write output if output file specified
            if (sink)
                WriteRawFrame(output_surface, sink);

            framenum++;
        }
    }

    printf("Decoded %d frames\n", framenum);

    fclose(sink);
    fclose(source);
    MFXVideoDECODE_Close(session);

    return 0;
}

// Read encoded stream from file
mfxStatus ReadEncodedStream(mfxBitstream &bs, mfxU32 codecid, FILE *f) {
    memmove(bs.Data, bs.Data + bs.DataOffset, bs.DataLength);
    bs.DataOffset = 0;
    bs.DataLength += static_cast<mfxU32>(
        fread(bs.Data + bs.DataLength, 1, bs.MaxLength - bs.DataLength, f));
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
int GetFreeSurfaceIndex(mfxFrameSurface1 *surface_pool, mfxU16 pool_size) {
    for (mfxU16 i = 0; i < pool_size; i++) {
        if (0 == surface_pool[i].Data.Locked)
            return i;
    }
    return MFX_ERR_NOT_FOUND;
}

char *ValidateFileName(char *in) {
    if (in) {
        if (strlen(in) > MAX_PATH)
            return NULL;
    }

    return in;
}
