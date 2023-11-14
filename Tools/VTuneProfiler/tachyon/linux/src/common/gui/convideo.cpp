//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright (C) Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================

#include "video.h"
#include <cassert>
#include <stdio.h>

unsigned int *              g_pImg = 0;
int                         g_sizex, g_sizey;
static video *g_video = 0;
static int g_fps = 0;

#if _WIN32 || _WIN64

static DWORD g_msec = 0;

#ifdef _WINDOWS
HINSTANCE video::win_hInstance = 0;
int video::win_iCmdShow = 0;
void video::win_set_class(WNDCLASSEX &wcex) { }
void video::win_load_accelerators(int idc)  { }
#endif //_WINDOWS

#else
#include <sched.h>
#include <sys/time.h>
struct timeval g_time;
#endif //_WIN32||_WIN64

#define CALC_FPS_ENABLED ((WINAPI_FAMILY != WINAPI_FAMILY_APP) && (!__ANDROID__))

video::video()
    // OpenGL* RGBA byte order for little-endian CPU
    : red_mask(0xff), red_shift(0), green_mask(0xff00),
      green_shift(8), blue_mask(0xff0000), blue_shift(16), depth(24)
{
    assert(g_video == 0);
    g_video = this; title = "Video"; updating = calc_fps = false;
}

bool video::init_window(int x, int y)
{
    g_sizex = x; g_sizey = y;
    g_pImg = new unsigned int[x*y];
    running = true;
    return false;
}

bool video::init_console()
{
    running = true;
    return true;
}

void video::terminate()
{
#if CALC_FPS_ENABLED
    if(calc_fps) {
        double fps = g_fps;
#if _WIN32 || _WIN64
        fps /= (GetTickCount()-g_msec)/1000.0;
#else
        struct timezone tz; struct timeval end_time; gettimeofday(&end_time, &tz);
        fps /= (end_time.tv_sec+1.0*end_time.tv_usec/1000000.0) - (g_time.tv_sec+1.0*g_time.tv_usec/1000000.0);
#endif
        printf("%s: %.1f fps\n", title, fps);
    }
#endif
    g_video = 0; running = false;
    if(g_pImg) { delete[] g_pImg; g_pImg = 0; }
}

video::~video()
{
    if(g_video) terminate();
}

//! Count and display FPS count in titlebar
bool video::next_frame()
{
#if CALC_FPS_ENABLED
    if(calc_fps){
        if(!g_fps) {
#if _WIN32 || _WIN64
            g_msec = GetTickCount();
#else
            struct timezone tz; gettimeofday(&g_time, &tz);
#endif
        }
        g_fps++;
    }
#endif
    return running;
}

//! Do standard loop
void video::main_loop()
{
    on_process();
}

//! Change window title
void video::show_title()
{
}

///////////////////////////////////////////// public methods of video class ///////////////////////

drawing_area::drawing_area(int x, int y, int sizex, int sizey)
    : start_x(x), start_y(y), size_x(sizex), size_y(sizey), pixel_depth(24),
    base_index(y*g_sizex + x), max_index(g_sizex*g_sizey), index_stride(g_sizex), ptr32(g_pImg)
{
    assert(x < g_sizex); assert(y < g_sizey);
    assert(x+sizex <= g_sizex); assert(y+sizey <= g_sizey);

    index = base_index; // current index
}

void drawing_area::update() {}
