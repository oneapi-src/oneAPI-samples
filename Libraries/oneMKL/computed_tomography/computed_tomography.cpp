//=====================================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// ====================================================================

/*
*  Content:
*      Reconstruct the original image from the Computed Tomography (CT)
*      data using oneMKL DPC++ fast Fourier transform (FFT) functions.
************************************************************************
* Usage:
* ======
*      program.out p q input.bmp radon.bmp restored.bmp
*      Input:
*      p, q - parameters of Radon transform
*      input.bmp - must be a 24-bit uncompressed input bitmap
*      Output:
*      radon.bmp - p-by-(2q+1) result of Radon transform of input.bmp
*      restored.bmp - 2q-by-2q result of FFT-based reconstruction
*
* Steps:
* ======
*      - Acquire Radon transforms from the original image - perform
*        line integrals for the original image in 'p' directions onto
*        '2q+1' points for each direction to obtain a sinogram.
*      Reconstruction phase -
*      1) Perform 'p' 1-D FFT's using oneMKL DFT DPC++ asynchronous USM API,
*         constructing full fourier representation of the object using 'p'
*         projections.
*      2) Interpolate from radial grid(fourier representation from step2)
*         onto Cartesian grid.
*      3) Perform one 2-D inverse FFT to obtain the reconstructed
*         image using oneMKL DFT DPCPP asynchronous USM API.
*/

#include <CL/sycl.hpp>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// For compatibility with beta09 -- not needed for new code.
#include "mkl_sycl.hpp"
#endif

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

#if !defined(REAL_DATA)
#define REAL_DATA double
#endif

typedef std::complex<REAL_DATA> complex;

typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                     oneapi::mkl::dft::domain::REAL>
        descriptor_real;
typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                     oneapi::mkl::dft::domain::COMPLEX>
        descriptor_complex;

// A simple transparent matrix class, row-major layout
template <typename T>
struct matrix {
    T *data;
    int h, w, ldw;
    sycl::queue q;
    matrix(sycl::queue &main_queue) : q{main_queue}, data{NULL} {}
    matrix(const matrix &);
    matrix &operator=(const matrix &);
    void deallocate()
    {
        if (data)
            free(data, q.get_context());
    }
    void allocate(int _h, int _w, int _ldw)
    {
        h   = _h;
        w   = _w;
        ldw = _ldw; // leading dimension for w
        deallocate();
        data = (T *)malloc_shared(sizeof(T) * h * ldw, q.get_device(), q.get_context());
    }
    ~matrix() { deallocate(); }
};
typedef matrix<REAL_DATA> matrix_r;
typedef matrix<complex> matrix_c;

// Computational functions
sycl::event step1_fft_1d(matrix_r &radon_image,
                         descriptor_real &fft1d,
                         sycl::queue &main_queue,
                         const sycl::vector_class<sycl::event> &deps = {});
sycl::event step2_interpolation(matrix_r &result,
                                matrix_r &radon_image,
                                sycl::queue &main_queue,
                                const sycl::vector_class<sycl::event> &deps = {});
sycl::event step3_ifft_2d(matrix_r &fhat,
                          descriptor_complex &ifft2d,
                          sycl::queue &main_queue,
                          const sycl::vector_class<sycl::event> &deps = {});

// Support functions
void bmp_read(matrix_r &image, std::string fname);
void bmp_write(std::string fname, const matrix_r &image, bool isComplex);
sycl::event acquire_radon(matrix_r &result, matrix_r &input, sycl::queue &main_queue);
template <typename T>
inline int is_odd(const T &n)
{
    return n & 1;
}
template <typename T>
void die(std::string err, T param);
void die(std::string err);

// Main function carrying out the steps mentioned above
int main(int argc, char **argv)
{
    int p = argc > 1 ? atoi(argv[1]) : 200; // # of projections in range 0..PI
    int q = argc > 2 ? atoi(argv[2]) : 100; // # of density points per projection is 2q+1

    std::string original_bmpname = argc > 3 ? argv[3] : "input.bmp";
    std::string radon_bmpname    = argc > 4 ? argv[4] : "radon.bmp";
    std::string restored_bmpname = argc > 5 ? argv[5] : "restored.bmp";

    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl;
            }
        }
    };

    // create execution queue with asynchronous error handling
    sycl::queue main_queue(sycl::device{sycl::default_selector{}}, exception_handler);

    std::cout << "Reading original image from " << original_bmpname << std::endl;
    matrix_r original_image(main_queue);
    bmp_read(original_image, original_bmpname);
    // bmpWrite( "check-original.bmp", originalImage ); //For debugging purpose

    std::cout << "Allocating radonImage for backprojection" << std::endl;
    matrix_r radon_image(main_queue);
    // space for p-by-(2q+2) reals, or for p-by-(q+1) complex numbers
    radon_image.allocate(p, 2 * q + 1, 2 * q + 2);
    if (!radon_image.data)
        die("cannot allocate memory for radonImage\n");

    std::cout << "Performing backprojection" << std::endl;
    auto ev = acquire_radon(radon_image, original_image, main_queue);
    ev.wait(); // wait here for bmpWrite
    bmp_write(radon_bmpname, radon_image, false);

    std::cout << "Restoring original: step1 - fft_1d in-place" << std::endl;
    descriptor_real fft1d((radon_image.w) - 1); // 2*q
    auto step1 = step1_fft_1d(radon_image, fft1d, main_queue);

    std::cout << "Allocating array for radial->cartesian interpolation" << std::endl;
    matrix_r fhat(main_queue);
    fhat.allocate(2 * q, 2 * 2 * q, 2 * 2 * q);
    if (!fhat.data)
        die("cannot allocate memory for fhat\n");

    std::cout << "Restoring original: step2 - interpolation" << std::endl;
    auto step2 = step2_interpolation(fhat, radon_image, main_queue, {step1});
    // step2.wait(); //wait here for bmpWrite
    // bmpWrite( "check-after-interpolation.bmp", fhat , true); //For debugging purpose

    std::cout << "Restoring original: step3 - ifft_2d in-place" << std::endl;
    descriptor_complex ifft2d({fhat.h, (fhat.w) / 2}); // fhat.w/2 in complex'es
    auto step3 = step3_ifft_2d(fhat, ifft2d, main_queue, {step2});

    std::cout << "Saving restored image to " << restored_bmpname << std::endl;
    step3.wait(); // Wait for the reconstructed image
    bmp_write(restored_bmpname, fhat, true);

    return 0;
}

// Step 1: batch of 1d r2c fft.
// ghat[j, lambda] <-- scale * FFT_1D( g[j,l] )
sycl::event step1_fft_1d(matrix_r &radon,
                         descriptor_real &fft1d,
                         sycl::queue &main_queue,
                         const sycl::vector_class<sycl::event> &deps)
{
    std::int64_t p   = radon.h;
    std::int64_t q2  = radon.w - 1; // w = 2*q + 1
    std::int64_t ldw = radon.ldw;
    REAL_DATA scale  = 1.0 / sqrt(0.0 + q2);

    // Make sure we can do in-place r2c
    if (is_odd(ldw))
        die("c-domain needs even ldw at line %i\n", __LINE__);
    if (q2 / 2 + 1 > ldw / 2)
        die("no space for in-place r2c, line %i\n", __LINE__);

    // Configure descriptor
    fft1d.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, p);
    fft1d.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, ldw);     // in REAL_DATA's
    fft1d.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, ldw / 2); // in complex'es
    fft1d.set_value(oneapi::mkl::dft::config_param::FORWARD_SCALE, scale);

    fft1d.commit(main_queue);

    auto fft1d_ev = oneapi::mkl::dft::compute_forward(fft1d, radon.data, deps);
    return fft1d_ev;
}

// Step 2: interpolation to Cartesian grid.
// ifreq_dom[x, y] <-- interpolation( freq_dom[theta, ksi] )
sycl::event step2_interpolation(matrix_r &fhat,
                                matrix_r &radon_image,
                                sycl::queue &main_queue,
                                const sycl::vector_class<sycl::event> &deps)
{
    // radonImage is the result of r2c FFT
    // rt(pp,:) contains frequences 0...q
    complex *rt = (complex *)radon_image.data;
    complex *ft = (complex *)fhat.data;

    int q   = (radon_image.w - 1) / 2; // w = 2q + 1
    int ldq = radon_image.ldw / 2;
    int p   = radon_image.h;

    int h   = fhat.h;
    int w   = fhat.w / 2;   // fhat.w/2 in complex'es
    int ldw = fhat.ldw / 2; // fhat.ldw/2 in complex'es

    auto ev = main_queue.submit([&](sycl::handler &cgh) {

        cgh.depends_on(deps);

        auto interpolateKernel = [=](sycl::item<1> item) {
            const int i = item.get_id(0);

            for (int j = 0; j < w; ++j) {
                REAL_DATA yy    = 2.0 * i / h - 1; // yy = [-1...1]
                REAL_DATA xx    = 2.0 * j / w - 1; // xx = [-1...1]
                REAL_DATA r     = sycl::sqrt(xx * xx + yy * yy);
                REAL_DATA phi   = sycl::atan2(yy, xx);
                complex fhat_ij = complex(0.);
                if (r <= 1) {
                    if (phi < 0) {
                        r = -r;
                        phi += M_PI;
                    }

                    int qq = sycl::floor(REAL_DATA(q + r * q + 0.5)) - q; // qq = [-q...q)
                    if (qq >= q)
                        qq = q - 1;

                    int pp = sycl::floor(REAL_DATA(phi / M_PI * p + 0.5)); // pp = [0...p)
                    if (pp >= p)
                        pp = p - 1;

                    if (qq >= 0)
                        fhat_ij = rt[pp * ldq + qq];
                    else
                        fhat_ij = std::conj(rt[pp * ldq - qq]);

                    if (is_odd(qq))
                        fhat_ij = -fhat_ij;
                    if (is_odd(i))
                        fhat_ij = -fhat_ij;
                    if (is_odd(j))
                        fhat_ij = -fhat_ij;
                }
                ft[i * ldw + j] = fhat_ij;
            }

        };

        cgh.parallel_for<class interpolateKernelClass>(sycl::range<1>(h), interpolateKernel);
    });

    return ev;
}

// Step 3: inverse FFT
// ifreq_dom[x, y] <-- IFFT_2D( ifreq_dom[x, y] )
sycl::event step3_ifft_2d(matrix_r &fhat,
                          descriptor_complex &ifft2d,
                          sycl::queue &main_queue,
                          const sycl::vector_class<sycl::event> &deps)
{
    // Configure descriptor
    std::int64_t strides[3] = {0, (fhat.ldw) / 2, 1}; // fhat.ldw/2, in complex'es
    ifft2d.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides);
    ifft2d.commit(main_queue);

    sycl::event ifft2d_ev = oneapi::mkl::dft::compute_backward(ifft2d, fhat.data, deps);

    return ifft2d_ev;
}

// Simplified BMP structure.
// See http://msdn.microsoft.com/en-us/library/dd183392(v=vs.85).aspx
#pragma pack(push, 1)
struct bmp_header {
    char bf_type[2];
    unsigned int bf_size;
    unsigned int bf_reserved;
    unsigned int bf_off_bits;

    unsigned int bi_size;
    unsigned int bi_width;
    unsigned int bi_height;
    unsigned short bi_planes;
    unsigned short bi_bit_count;
    unsigned int bi_compression;
    unsigned int bi_size_image;
    unsigned int bi_x_pels_per_meter;
    unsigned int bi_y_pels_per_meter;
    unsigned int bi_clr_used;
    unsigned int bi_clr_important;
};
#pragma pack(pop)

// Read image from fname and convert it to gray-scale REAL.
void bmp_read(matrix_r &image, std::string fname)
{
    std::fstream fp;
    fp.open(fname, std::fstream::in | std::fstream::binary);
    if (fp.fail())
        die("cannot open the file %s\n", fname);

    bmp_header header;

    fp.read((char *)(&header), sizeof(header));
    if (header.bi_bit_count != 24)
        die("not a 24-bit image in %s\n", fname);
    if (header.bi_compression)
        die("%s is compressed bmp\n", fname);

    image.allocate(header.bi_height, header.bi_width, header.bi_width);
    if (!image.data)
        die("no memory to read %s\n", fname);

    fp.seekg(sizeof(header), std::ios_base::beg);

    REAL_DATA *image_data = (REAL_DATA *)image.data;

    for (int i = 0; i < image.h; ++i) {
        for (int j = 0; j < image.w; ++j) {
            struct {
                unsigned char b, g, r;
            } pixel;
            fp.read((char *)(&pixel), 3);
            REAL_DATA gray               = (255 * 3.0 - pixel.r - pixel.g - pixel.b) / 255;
            image_data[i * image.ldw + j] = gray;
        }
        fp.seekg((4 - 3 * image.w % 4) % 4, std::ios_base::cur);
    }
    fp.close();
}

inline REAL_DATA to_real(const REAL_DATA &x)
{
    return x;
}
inline REAL_DATA to_real(const complex &x)
{
    return std::abs(x);
}

template <typename T>
void bmp_write_templ(std::string fname, int h, int w, int ldw, T *data)
{
    unsigned sizeof_line  = (w * 3 + 3) / 4 * 4;
    unsigned sizeof_image = h * sizeof_line;

    bmp_header header = {{'B', 'M'},
                        unsigned(sizeof(header) + sizeof_image),
                        0,
                        sizeof(header),
                        sizeof(header) - offsetof(bmp_header, bi_size),
                        unsigned(w),
                        unsigned(h),
                        1,
                        24,
                        0,
                        sizeof_image,
                        6000,
                        6000,
                        0,
                        0};

    std::fstream fp;
    fp.open(fname, std::fstream::out | std::fstream::binary);
    if (fp.fail())
        die("failed to save the image, cannot open file", fname);

    fp.write((char *)(&header), sizeof(header));

    REAL_DATA minabs = 1e38, maxabs = 0;
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j) {
            REAL_DATA ijabs = to_real(data[i * ldw + j]);
            if (!(ijabs > minabs))
                minabs = ijabs;
            if (!(ijabs < maxabs))
                maxabs = ijabs;
        }

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            REAL_DATA ijabs = to_real(data[i * ldw + j]);
            REAL_DATA gray  = 255 * (ijabs - maxabs) / (minabs - maxabs);

            struct {
                unsigned char b, g, r;
            } pixel;
            pixel.b = pixel.g = pixel.r = (unsigned char)(gray + 0.5);
            fp.write((char *)(&pixel), 3);
        }
        for (int j = 3 * w; j % 4; ++j)
            fp.put(0);
    }
    fp.close();
}

void bmp_write(std::string fname, const matrix_r &image, bool isComplex)
{
    if (isComplex)
        bmp_write_templ(fname, image.h, image.w / 2, image.ldw / 2, (complex *)image.data);
    else
        bmp_write_templ(fname, image.h, image.w, image.ldw, image.data);
}

sycl::event acquire_radon(matrix_r &result, matrix_r &input, sycl::queue &main_queue)
{

    int h_r = result.h, w_r = result.w, ldw_r = result.ldw;
    int h_i = input.h, w_i = input.w, ldw_i = input.ldw;

    // Integrate image along line [(x,y),(cos theta, sin theta)] = R*s
    auto integrate_along_line = [=](REAL_DATA theta, REAL_DATA s, int h, int w, int ldw,
                                  REAL_DATA *data) {
        REAL_DATA R = 0.5 * sycl::sqrt(REAL_DATA(h * h + w * w));
        REAL_DATA S = s * R, B = sycl::sqrt(1 - s * s) * R;
        REAL_DATA cs = sycl::cos(theta), sn = sycl::sin(theta);
        REAL_DATA dl = 1, dx = -dl * sn, dy = dl * cs; // integration step
        // unadjusted start of integration
        REAL_DATA x0 = 0.5 * w + S * cs + B * sn;
        REAL_DATA y0 = 0.5 * h + S * sn - B * cs; // unadjusted end of integration
        REAL_DATA x1 = 0.5 * w + S * cs - B * sn;
        REAL_DATA y1 = 0.5 * h + S * sn + B * cs;

        int N = 0; // number of sampling points on the interval
        do {
            // Adjust start-end of the integration interval
            if (x0 < 0) {
                if (x1 < 0)
                    break;
                else {
                    y0 -= (0 - x0) * cs / sn;
                    x0 = 0;
                }
            }
            if (y0 < 0) {
                if (y1 < 0)
                    break;
                else {
                    x0 -= (0 - y0) * sn / cs;
                    y0 = 0;
                }
            }
            if (x1 < 0) {
                if (x0 < 0)
                    break;
                else {
                    y1 -= (0 - x1) * cs / sn;
                    x1 = 0;
                }
            }
            if (y1 < 0) {
                if (y0 < 0)
                    break;
                else {
                    x1 -= (0 - y1) * sn / cs;
                    y1 = 0;
                }
            }
            if (x0 > w) {
                if (x1 > w)
                    break;
                else {
                    y0 -= (w - x0) * cs / sn;
                    x0 = w;
                }
            }
            if (y0 > h) {
                if (y1 > h)
                    break;
                else {
                    x0 -= (h - y0) * sn / cs;
                    y0 = h;
                }
            }
            if (x1 > w) {
                if (x0 > w)
                    break;
                else {
                    y1 -= (w - x1) * cs / sn;
                    x1 = w;
                }
            }
            if (y1 > h) {
                if (y0 > h)
                    break;
                else {
                    x1 -= (h - y1) * sn / cs;
                    y1 = h;
                }
            }
            // Compute number of steps
            N = int(sycl::fabs(dx) > sycl::fabs(dy) ? ((x1 - x0) / dx) : ((y1 - y0) / dy));
        } while (0);

        // Integrate
        REAL_DATA sum = 0;
        for (int n = 0; n < N; ++n) {
            int i = sycl::floor(y0 + n * dy + 0.5 * dy);
            int j = sycl::floor(x0 + n * dx + 0.5 * dx);
            sum += data[i * ldw + j];
        }
        sum *= dl;

        return sum;
    };

    REAL_DATA *input_data  = (REAL_DATA *)input.data;
    REAL_DATA *result_data = (REAL_DATA *)result.data;

    sycl::event ev = main_queue.submit([&](sycl::handler &cgh) {

        auto acquire_radon_kernel = [=](sycl::item<1> item) {
            const int i = item.get_id(0);

            REAL_DATA theta = i * M_PI / h_r; // theta=[0,...,M_PI)
            for (int j = 0; j < w_r; ++j) {
                REAL_DATA s          = -1. + (2.0 * j + 1) / w_r; // s=(-1,...,1)
                REAL_DATA projection = integrate_along_line(theta, s, h_i, w_i, ldw_i, input_data);
                result_data[i * ldw_r + j] = projection;
            }

        };

        cgh.parallel_for<class acquire_radon_class>(sycl::range<1>(h_r), acquire_radon_kernel);
    });

    return ev;
}

template <typename T>
void die(std::string err, T param)
{
    std::cout << "Fatal error: " << err << " " << param << std::endl;
    fflush(0);
    exit(1);
}

void die(std::string err)
{
    std::cout << "Fatal error: " << err << " " << std::endl;
    fflush(0);
    exit(1);
}
