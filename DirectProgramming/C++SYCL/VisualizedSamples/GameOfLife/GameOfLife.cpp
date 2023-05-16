#include <SDL2/SDL.h>
#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>
#include "GoL.hpp"

using namespace sycl;
using namespace std;

// Window height and width
constexpr int windowWidth = 640;
constexpr int windowHeight = 480;

void ShowDevice(queue& q) {
	// Output platform and device information.
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

// Handle SDL events while animation phase
std::pair<bool, bool> handleEvents(const SDL_Event& event, GameOfLife& gol) {
	static bool isMousePressed = false;
	static bool isSpacePressed = false;
	switch (event.type) {
	case SDL_QUIT:
		return { true, false };
	case SDL_KEYDOWN:
	{
		if (event.key.keysym.sym == SDLK_ESCAPE)
			return { true, false };
		if (event.key.keysym.sym == SDLK_SPACE && !isSpacePressed)
		{
			isSpacePressed = true;
			return { false, true };
		}
	}
	case SDL_KEYUP:
	{
		if (event.key.keysym.sym == SDLK_SPACE && isSpacePressed)
			isSpacePressed = false;
		return { false, false };
	}
	case SDL_MOUSEBUTTONDOWN:
	{
		if (event.button.button == SDL_BUTTON_RIGHT)
		{
			isMousePressed = true;
		}
		return { false, false };
	}
	case SDL_MOUSEBUTTONUP:
	{
		isMousePressed = false;
		return { false, false };
	}
	case SDL_MOUSEMOTION:
	{
		if (isMousePressed) {
			// Convert relative movment of the mouse from pixels to 0.0 - 1.0 range
			double x = event.motion.xrel / (double)windowWidth;
			double y = event.motion.yrel / (double)windowHeight;
			gol.pan(x, y);
		}
		return { false, false };
	}
	case SDL_MOUSEWHEEL:
	{
		gol.scale(event.wheel.preciseY);
		return { false, false };
	}
	}
	return { false, false };
}

// Handle additonal event during placement phase
std::pair<bool, bool> handlePlaceEvents(const SDL_Event& event, GameOfLife& gol)
{
	if (event.button.button == SDL_BUTTON_LEFT && event.type == SDL_MOUSEBUTTONDOWN)
	{
		double x = event.button.x / (double)windowWidth;
		double y = event.button.y / (double)windowHeight;
		gol.modifyCell(x, y);
	}
	return handleEvents(event, gol);
}



int wmain()
{
	// width and height of the texture
	int width = 256;
	int height = 256;
	// How much cells should be alive in percent.
	int probability = 10;

	// Initialize SDL and create requaired structs.
	SDL_Init(SDL_INIT_VIDEO);
	auto window = SDL_CreateWindow("GameOfLife", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowWidth, windowHeight, 0);
	auto renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	auto texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, width, height);

	// Calculate number of bytes between each row of pixels.
	int pitch = sizeof(uint32_t) * width;

	queue q{};
	ShowDevice(q);

	// Create Game of Life class with width x height.
	GameOfLife gol{ width, height, probability, q};

	bool quit = false;
	bool nextFrame = false;
	SDL_Event event;

	// Placement phase
	while (!quit)
	{
		// Wait for SDL event.
		SDL_WaitEvent(&event);
		std::tie(quit, nextFrame) = handlePlaceEvents(event, gol);

		// Update texture after event and render it.
		SDL_UpdateTexture(texture, NULL, gol.getCells(), pitch);
		SDL_RenderCopy(renderer, texture, gol.getTextureRect(), NULL);

		SDL_RenderPresent(renderer);
	}

	// Copy data to device
	gol.copyDataToDevice();
	uint32_t* pixels;

	quit = false;
	while (!quit)
	{
		// Wait for SDL event.
		SDL_WaitEvent(&event);
		std::tie(quit, nextFrame) = handleEvents(event, gol);
		if (nextFrame)
		{
			// Lock texture and update the pixels pointer from where SDL will read data.
			SDL_LockTexture(texture, NULL, (void**)&pixels, &pitch);
			// Calculate the next step.
			gol.calculateNextStep(pixels);
			// Unlock the texture.
			SDL_UnlockTexture(texture);
		}
		// Copy texture to renderer.
		SDL_RenderCopy(renderer, texture, gol.getTextureRect(), NULL);
		// Render present renderer.
		SDL_RenderPresent(renderer);
	}

	// Destroy window and clean the SDL.
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}
