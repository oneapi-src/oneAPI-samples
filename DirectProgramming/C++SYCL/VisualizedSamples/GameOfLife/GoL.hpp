#pragma once
#include <sycl/sycl.hpp>
#include <cstdlib>
#include <ctime>

using namespace sycl;
using namespace std;

class GameOfLife {
public:
	GameOfLife(int width, int height, int probability, queue& q);	
	void modifyCell(double x, double y);

	void pan(double x, double y);
	void scale(double scale);

	const uint32_t* getCells() { return cells; }
	uint32_t* getCellsFromDevice();
	void copyDataToDevice();
	const SDL_Rect* getTextureRect() { return &textureRect; }
	void calculateNextStep(uint32_t* pixels);
	
private:
	const int width;
	const int height;
	uint32_t* cells;
	uint32_t* cells_device;
	SDL_Rect textureRect;

	queue q;
	event e;
};

void GameOfLife::copyDataToDevice()
{
	q.copy<uint32_t>(cells, cells_device, width * height).wait();
}

uint32_t* GameOfLife::getCellsFromDevice()
{
	q.copy<uint32_t>(cells_device, cells, width * height, e).wait();
	return cells;
}

void GameOfLife::calculateNextStep(uint32_t* pixels)
{
	auto width = this->width;
	auto height = this->height;
	auto cells_device = this->cells_device;

	e = q.submit([&](handler& h) {
		// Iterate over image and calculate if cell will live.
		h.parallel_for(range<2>(height - 2, width - 2), [=](auto index)
			{
				int y = index[0] + 1;
				int x = index[1] + 1;
				auto idx = y * width + x;
				auto value = 0;
				value += cells_device[idx + 1] & 1;
				value += cells_device[idx - 1] & 1;
				value += cells_device[idx + width] & 1;
				value += cells_device[idx - width] & 1;
				value += cells_device[idx + 1 + width] & 1;
				value += cells_device[idx + 1 - width] & 1;
				value += cells_device[idx - 1 + width] & 1;
				value += cells_device[idx - 1 - width] & 1;
				if (cells_device[idx] && (value < 2 || value > 3))
					cells_device[idx] = 0x0;
				if (value == 3)
					cells_device[idx] = 0xffffffff;
			});
		});
	// Copy data to SDL buffer.
	q.copy<uint32_t>(cells_device, pixels, width * height, e).wait();
}

GameOfLife::GameOfLife(int width, int height, int probability, queue& q) :
	width(width), height(height), q(q)
{
	// Allocate memory for cells on host
	cells = (uint32_t*)calloc(width * height, sizeof(uint32_t));
	// Allocate memory for cells on device
	cells_device = sycl::malloc_device<uint32_t>(width * height, q);
	
	// Fill randomly cells with 10% probability
	for (auto y = 1; y < height - 1; y++)
	{
		for (auto x = 1; x < width - 1; x++)
		{
			if(rand() % 100 < probability) 
			 cells[y*width + x] = 0xffffffff;
		}
	}
	textureRect = SDL_Rect{ 0, 0, width, height };
}

void GameOfLife::modifyCell(double x, double y) 
{
	// Calculate which cell needs to be modified 
	int idx_x = textureRect.x + x * textureRect.w;
	int idx_y = textureRect.y + y * textureRect.h;
	
	// Flip the value 
	cells[idx_y * width + idx_x] = ~cells[idx_y * width + idx_x];
}


void GameOfLife::scale(double scale)
{
	// calculate new sizes after scaling
	int newWidth = textureRect.w * (1.0 + 0.05 * scale);
	int widthDiff = (newWidth - textureRect.w);
	int newX = textureRect.x + widthDiff/2;
	int newW = textureRect.w - widthDiff;


	int newHeight = textureRect.h * (1.0 + 0.05 * scale);
	int heightDiff = (newHeight  - textureRect.h);
	int newY = textureRect.y + heightDiff/2;
	int newH = textureRect.h - heightDiff;

	if (newX < 0)
		newX = 0;
	if (newW < 60)
		newW = 60;
	if (newW > width)
		newW = width;
	if (newX + newW > width)
		newX = width - newW;

	if (newY < 0)
		newY = 0;
	if (newH < 60)
		newH = 60;
	if (newH > height)
		newH = height;
	if (newY + newH > height)
		newY = height - newH;

	textureRect = SDL_Rect{ newX, newY, newW, newH };
}

void GameOfLife::pan(double x, double y)
{
	int xOffset = textureRect.w * x;
	if ((textureRect.x + textureRect.w - xOffset < width) &&
		(textureRect.x - xOffset > 0))
	{
		textureRect.x -= xOffset;
	}
	int yOffset = textureRect.h * y;
	if ((textureRect.y + textureRect.h - yOffset < height) &&
		(textureRect.y - yOffset > 0))
	{
		textureRect.y -= yOffset;
	}
}



