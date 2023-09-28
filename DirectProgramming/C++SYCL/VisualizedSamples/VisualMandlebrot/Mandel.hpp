#pragma once
#include <complex>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

std::complex<double> complex_square(std::complex<double> c)
{
	return std::complex<double>(c.real() * c.real() - c.imag() * c.imag(), c.real() * c.imag() * 2);
}

struct MandelParameters {
	int width;
	int height;
	int maxIterations;

	double xmin;
	double xmax;
	double ymin;
	double ymax;

	using ComplexF = std::complex<double>;

	MandelParameters(int width, int height, int maxIterations, double xmin, double xmax, double ymin, double ymax) :
		width(width),
		height(height),
		maxIterations(maxIterations),
		xmin(xmin),
		xmax(xmax),
		ymin(ymin),
		ymax(ymax) {}

	// Scale from 0..width to xmin..xmax
	double ScaleRow(int i) const { return xmin + (i * (xmax - xmin) / width); }

	// Scale from 0..height to ymin..ymax
	double ScaleCol(int i) const { return -(ymin + (i * (ymax - ymin) / height)); }

	// Mandelbrot set are points that do not diverge within max_iterations.
	int Point(const ComplexF& c) const {
		int count = 0;
		ComplexF z = 0;

		for (int i = 0; i < maxIterations; ++i) {
			auto r = z.real();
			auto im = z.imag();

			// Leave loop if diverging.
			if (((r * r) + (im * im)) >= 4.0f) {
				break;
			}

			// z = z * z + c;
			z = complex_square(z) + c;
			count++;
		}

		return count;
	}
};


class Mandelbrot {
public:
	Mandelbrot(int width, int height, int maxIterations, double xmin, double xmax, double ymin, double ymax, queue& q);
	void Calculate(uint32_t* pixels);
	MandelParameters getParameters() const { return parameters; }
	void scale(double xoffset, double yoffset, double scale);
	void pan(double xoffset, double yoffset);

private:
	queue& q;
	MandelParameters parameters;
};

Mandelbrot::Mandelbrot(int width, int height, int maxIterations, double xmin, double xmax, double ymin, double ymax, queue& q) :
	q(q),
	parameters(width, height, maxIterations, xmin, xmax, ymin, ymax)
{
}

void Mandelbrot::Calculate(uint32_t* pixels) {

	MandelParameters p = getParameters();
	const int width = p.width;
	const int height = p.height;
	const int maxIterations = p.maxIterations;
	buffer pixelsBuf(pixels, range(width * height));

	// We submit a command group to the queue.
	q.submit([&](handler& h) {

		accessor ldata(pixelsBuf, h, write_only, no_init);
		// Iterate over image and compute mandel for each point.
	    h.parallel_for(range<1>(height * width), [=](auto index) {
	    	int y = index / height;
	    	int x = index % height;
	    	auto c = std::complex<double>(p.ScaleRow(x), p.ScaleCol(y));
	    	int value = p.Point(c);
	    	double normalized = (1.0 * value) / maxIterations;
			ldata[index] = uint32_t(normalized * 0xFFFFFF);
	    	ldata[index] <<= 8;
	    	ldata[index] |= 0xFF;
		});
    }).wait();	
}

void Mandelbrot::scale(double xoffset, double yoffset, double scale)
{
	// calculate cursour position in the mandelbrot space
	double x = parameters.xmin + (parameters.xmax - parameters.xmin) * xoffset;
	double y = parameters.ymin + (parameters.ymax - parameters.ymin) * yoffset;

	// scale the space
	parameters.xmin *= 1 + 0.02 * scale;
	parameters.xmax *= 1 + 0.02 * scale;
	parameters.ymin *= 1 + 0.02 * scale;
	parameters.ymax *= 1 + 0.02 * scale;

	// calculate coursor position in the scaled mandelbrot space
	double x2 = parameters.xmin + (parameters.xmax - parameters.xmin) * xoffset;
	double y2 = parameters.ymin + (parameters.ymax - parameters.ymin) * yoffset;

	// calculate the offset between position before and after scaling
	double offset_x = x2 - x;
	double offset_y = y2 - y;

	// move the space by offset
	parameters.xmin -= offset_x;
	parameters.xmax -= offset_x;
	parameters.ymin -= offset_y;
	parameters.ymax -= offset_y;
}

void Mandelbrot::pan(double xoffset, double yoffset)
{
	// convert the camera movment from 0 - 1.0 range to distance in mandelbrot space. 
	double w = (parameters.xmax - parameters.xmin) * xoffset;
	double h = (parameters.ymax - parameters.ymin) * yoffset;

	// move the space by offset
	parameters.xmin -= w;
	parameters.xmax -= w;
	parameters.ymin -= h;
	parameters.ymax -= h;
}

