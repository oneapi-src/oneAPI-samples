#include <SDL2/SDL.h>
#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>
#include "Mandel.hpp"

using namespace sycl;
using namespace std;

constexpr int windowWidth = 640;
constexpr int windowHeight = 480;

void ShowDevice(queue& q) {
	// Print device info.
	auto device = q.get_device();
	auto p_name = device.get_platform().get_info<info::platform::name>();
	cout << std::setw(20) << "Platform Name: " << p_name << "\n";
	auto p_version = device.get_platform().get_info<info::platform::version>();
	cout << std::setw(20) << "Platform Version: " << p_version << "\n";
	auto d_name = device.get_info<info::device::name>();
	cout << std::setw(20) << "Device Name: " << d_name << "\n";
	auto max_work_group = device.get_info<info::device::max_work_group_size>();
	cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
	auto max_compute_units = device.get_info<info::device::max_compute_units>();
	cout << std::setw(20) << "Max Compute Units: " << max_compute_units << "\n\n";
}

// Function for handling SDL events.
bool handleEvents(const SDL_Event& event, Mandelbrot& mandel)
{
	static bool isMousePressed = false;
	switch (event.type) {
	case SDL_QUIT: 
		return true;
	case SDL_MOUSEBUTTONDOWN:
		isMousePressed = true;
		return false;
	case SDL_MOUSEBUTTONUP:
		isMousePressed = false;
		return false;
	case SDL_MOUSEMOTION:
		if(isMousePressed)
		{
			// Convert relative movment of the mouse from pixels to 0.0 - 1.0 range
			double x = event.motion.xrel / (double)windowWidth;
			double y = event.motion.yrel / (double)windowHeight;
			mandel.pan(x, y);
		}
		return false;
	case SDL_MOUSEWHEEL:
		// Get the mouse position on screen in pixels and map it to 0.0 - 1.0 range
		int posx, posy;
		SDL_GetMouseState(&posx, &posy);
		double x = posx / (double)windowWidth;
		double y = posy / (double)windowHeight;
		mandel.scale(x, y, -event.wheel.preciseY);
		return false;
	}
	return false;
}

#ifdef _WIN32
int wmain()
#elif __linux__
int main()
#endif
{
	// Texture size
	int width = 1024;
	int height = 1024;
	int maxIterations = 50;

	// Initialize SDL and create requaired structs.
	SDL_Init(SDL_INIT_VIDEO);
	auto window = SDL_CreateWindow("Mandelbrot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowWidth, windowHeight, 0);
	auto renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	auto texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, width, height);

	// Calculate number of bytes between each row of pixels.
	int pitch = sizeof(uint32_t) * width;

	// Create Queue with default selector.
	queue q(default_selector_v);
	ShowDevice(q);


	// Create mandelbrot class with width x height texture size, maxIterations, and rendered range from -2 to 2 in X and Y direction
	Mandelbrot mandelbrot(width, height, maxIterations, -2, 2, -2, 2, q);
	uint32_t* pixels;
	bool quit = false;
	SDL_Event event;

	while (!quit)
	{
		// Wait for SDL event.
		SDL_WaitEvent(&event);
		quit = handleEvents(event, mandelbrot);

		// Lock texture and update the pixels pointer from where SDL will read data.
		SDL_LockTexture(texture, NULL, (void**)&pixels, &pitch);
		// Calculate mandelbrot and write pixels to pixel pointer
		mandelbrot.Calculate(pixels);
		// Unlock the texture.
		SDL_UnlockTexture(texture);
		// Copy texture to renderer.
		SDL_RenderCopy(renderer, texture, NULL, NULL);
		// Render present renderer.
		SDL_RenderPresent(renderer);
	}

	// Destroy window and clean the SDL.
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}
