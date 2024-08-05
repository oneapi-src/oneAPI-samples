#pragma once
#include <complex>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

template <typename T> std::complex<T> complex_square(std::complex<T> c)
{
	return std::complex<T>(c.real() * c.real() - c.imag() * c.imag(), c.real() * c.imag() * 2);
}

template <typename T> class MandelParameters {
	using ComplexF = std::complex<T>;

	public:
		MandelParameters() {}
    	MandelParameters(int width, int height, int maxIterations, T xmin, T xmax, T ymin, T ymax):
			width(width),
			height(height),
			maxIterations(maxIterations),
			xmin(xmin),
			xmax(xmax),
			ymin(ymin),
			ymax(ymax) {}

		T ScaleRow(int i) const { return xmin + (i * (xmax - xmin) / width); }

		// Scale from 0..height to ymin..ymax
		T ScaleCol(int i) const { return -(ymin + (i * (ymax - ymin) / height)); }

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
				z = complex_square<T>(z) + c;
				count++;
			}

			return count;
		}

		void scale(T xoffset, T yoffset, T scale) {
			// calculate cursour position in the mandelbrot space
			T x = xmin + (xmax - xmin) * xoffset;
			T y = ymin + (ymax - ymin) * yoffset;

			// scale the space
			xmin *= 1 + 0.02 * scale;
			xmax *= 1 + 0.02 * scale;
			ymin *= 1 + 0.02 * scale;
			ymax *= 1 + 0.02 * scale;

			// calculate coursor position in the scaled mandelbrot space
			T x2 = xmin + (xmax - xmin) * xoffset;
			T y2 = ymin + (ymax - ymin) * yoffset;

			// calculate the offset between position before and after scaling
			T offset_x = x2 - x;
			T offset_y = y2 - y;

			// move the space by offset
			xmin -= offset_x;
			xmax -= offset_x;
			ymin -= offset_y;
			ymax -= offset_y;
		}

		void pan(T xoffset, T yoffset) {
			// convert the camera movment from 0 - 1.0 range to distance in mandelbrot space.
			T w = (xmax - xmin) * xoffset;
			T h = (ymax - ymin) * yoffset;

			// move the space by offset
			xmin -= w;
			xmax -= w;
			ymin -= h;
			ymax -= h;
		}

		int width;
		int height;
		int maxIterations;
		T xmin;
		T xmax;
		T ymin;
		T ymax;
};


class Mandelbrot {
public:
	Mandelbrot(int width, int height, int maxIterations, double xmin, double xmax, double ymin, double ymax, queue& q);
	Mandelbrot(int width, int height, int maxIterations, float xmin, float xmax, float ymin, float ymax, queue& q);
	void Calculate(uint32_t* pixels);
	void scale(double xoffset, double yoffset, double scale);
	void pan(double xoffset, double yoffset);
	MandelParameters<float> getSPParameters() const { return sp_parameters; }
	MandelParameters<double> getDPParameters() const { return dp_parameters; }

private:
	void CalculateSP(uint32_t* pixels);
	void CalculateDP(uint32_t* pixels);
	queue& q;
	MandelParameters<float> sp_parameters;
	MandelParameters<double> dp_parameters;
	bool singlePrecision;
};

Mandelbrot::Mandelbrot(int width, int height, int maxIterations, double xmin, double xmax, double ymin, double ymax, queue& q) :
	q(q),
	dp_parameters(width, height, maxIterations, xmin, xmax, ymin, ymax),
	singlePrecision(false)
{
}

Mandelbrot::Mandelbrot(int width, int height, int maxIterations, float xmin, float xmax, float ymin, float ymax, queue& q) :
	q(q),
	sp_parameters(width, height, maxIterations, xmin, xmax, ymin, ymax),
	singlePrecision(true)
{
}

void Mandelbrot::Calculate(uint32_t* pixels) {

	if (singlePrecision)
		CalculateSP(pixels);
	else
		CalculateDP(pixels);
}

void Mandelbrot::CalculateSP(uint32_t* pixels) {
	MandelParameters<float> parameters = getSPParameters();
	const int width = parameters.width;
	const int height = parameters.height;
	const int maxIterations = parameters.maxIterations;
	buffer pixelsBuf(pixels, range(width * height));

	// We submit a command group to the queue.
	q.submit([&](handler& h) {

		accessor ldata(pixelsBuf, h, write_only, no_init);
		// Iterate over image and compute mandel for each point.
	    h.parallel_for(range<1>(height * width), [=](auto index) {
	    	int y = index / height;
	    	int x = index % height;
	    	auto c = std::complex<float>(parameters.ScaleRow(x), parameters.ScaleCol(y));
	    	int value = parameters.Point(c);
	    	float normalized = (1.0f * value) / maxIterations;
			ldata[index] = uint32_t(normalized * 0xFFFFFF);
	    	ldata[index] <<= 8;
	    	ldata[index] |= 0xFF;
		});
    }).wait();
}

void Mandelbrot::CalculateDP(uint32_t* pixels) {
	MandelParameters<double> parameters = getDPParameters();
	const int width = parameters.width;
	const int height = parameters.height;
	const int maxIterations = parameters.maxIterations;
	buffer pixelsBuf(pixels, range(width * height));

	// We submit a command group to the queue.
	q.submit([&](handler& h) {

		accessor ldata(pixelsBuf, h, write_only, no_init);
		// Iterate over image and compute mandel for each point.
	    h.parallel_for(range<1>(height * width), [=](auto index) {
	    	int y = index / height;
	    	int x = index % height;
	    	auto c = std::complex<double>(parameters.ScaleRow(x), parameters.ScaleCol(y));
	    	int value = parameters.Point(c);
	    	double normalized = (1.0 * value) / maxIterations;
			ldata[index] = uint32_t(normalized * 0xFFFFFF);
	    	ldata[index] <<= 8;
	    	ldata[index] |= 0xFF;
		});
    }).wait();	
}

void Mandelbrot::scale(double xoffset, double yoffset, double scale)
{
	if (singlePrecision)
		sp_parameters.scale((float) xoffset, (float) yoffset, (float) scale);
	else
		dp_parameters.scale(xoffset, yoffset, scale);
}

void Mandelbrot::pan(double xoffset, double yoffset)
{
	if (singlePrecision)
		sp_parameters.pan((float) xoffset, (float) yoffset);
	else
		dp_parameters.pan(xoffset, yoffset);
}

