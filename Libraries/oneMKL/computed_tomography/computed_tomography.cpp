//=====================================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// ====================================================================

/*
*  Content:
*      Reconstruct the original image from the Computed Tomography (CT)
*      data using oneMKL DPC++ Discrete Fourier Transforms (DFTs).
************************************************************************
*/
#include <string>
constexpr int default_p = 400;
constexpr int default_q = 400;
constexpr double default_S_to_D = 1.0;
static const std::string default_original_bmpname = "input.bmp";
static const std::string default_radon_bmpname = "radon.bmp";
static const std::string default_restored_bmpname = "restored.bmp";
static const std::string default_errors_bmpname = "errors.bmp";
#ifdef _WIN64
static const std::string program_name = "compute_tomography.exe";
#else
static const std::string program_name = "compute_tomography";
#endif
constexpr double arbitrary_error_threshold = 0.1;
static const std::string usage_info =
"\n\
  Usage:\n\
  ======\n\
    " + program_name + " p q S_to_D in radon_out restored_out err_out\n\
  Inputs:\n\
  -------\n\
  p             - number of projection directions considered for the Radon\n\
                  transform (number of angles spanning a range of M_PI).\n\
                  p must be a strictly positive integer (default value is " +
                  std::to_string(default_p) + ").\n\
  q             - number of samples of the Radon transform for every\n\
                  projection direction.\n\
                  q must be a strictly positive integer (default value is " +
                  std::to_string(default_q) + ").\n\
  S_to_D        - ratio of the scanning width used for sampling the Radon\n\
                  transform (for any projection direction) to the diagonal of\n\
                  the input image.\n\
                  S_to_D must be a floating-point value larger than or equal\n\
                  to 1.0 (default value is " +
                  std::to_string(default_S_to_D) + ").\n\
  in            - name of the input image used to generate Radon transform data.\n\
                  The file must be a 24-bit uncompressed bitmap file.\n\
                  \"" + default_original_bmpname + "\" is considered by default.\n\
  Outputs:\n\
  --------\n\
  radon_out     - name of a 24-bit uncompressed bitmap image file storing a\n\
                  gray-scaled image representing the generated Radon\n\
                  transform data points.\n\
                  \"" + default_radon_bmpname + "\" is used by default.\n\
  restored_out  - name of a 24-bit uncompressed bitmap image file storing a\n\
                  gray-scaled image representing the image reconstructed\n\
                  from the Radon transform data points.\n\
                  \"" + default_restored_bmpname + "\" is used by default.\n\
  err_out       - name of a 24-bit uncompressed bitmap image file storing a\n\
                  gray-scaled image representing the pixel-wise error in the\n\
                  restored_out file. This file is created only if the mean\n\
                  global error is found to exceed "
                  + std::to_string(100.0*arbitrary_error_threshold) +
                  "% of the maximum\n\
                  gray-scale value in the original image.\n\
                  \"" + default_errors_bmpname + "\" is considered by default.\n\
";

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>

#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

namespace dft_ns = oneapi::mkl::dft;
using real_descriptor_t =
    dft_ns::descriptor<dft_ns::precision::DOUBLE, dft_ns::domain::REAL>;
using complex_t = std::complex<double>;
// This sample takes the side length of the input image's pixels as unit length
constexpr double in_pix_len = 1.0;

// A transparent matrix structure accouting for minimal padding as required for
// - either a batch of h real 1D in-place DFTs of length w;
// - or one real 2D in-place DFT of lengths {h, w}.
// For such operations (real in-place DFTs), data padding is required to store
// all the backward domain's elements.
struct padded_matrix {
    sycl::queue q;
    double *data;
    int h, w;
    padded_matrix(sycl::queue &main_queue) :
        q{main_queue}, data{nullptr}, h(0), w(0) {}
    padded_matrix(const padded_matrix &) = delete;
    padded_matrix &operator=(const padded_matrix &) = delete;
    void deallocate() {
        if (data)
            sycl::free(data, q);
        data = nullptr;
        h = w = 0;
    }
    inline int ldw() const {
        return 2 * (w / 2 + 1);
    }
    void allocate(int _h, int _w) {
        deallocate();
        h   = _h;
        w   = _w;
        data = (double *)sycl::malloc_shared(sizeof(double) * h * ldw(), q);
    }
    ~padded_matrix() {
        deallocate();
    }
};
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
struct pixel {
    unsigned char b, g, r;
};
// Routine terminating the application and reporting ad-hoc information.
void die(const std::string& err) {
    std::cerr << "Fatal error: " << err << " " << std::endl
              << std::flush;
    std::exit(EXIT_FAILURE);
}
// Routine reading a 24-bit uncompressed bitmap image file and converting it to
// a gray-scale padded_matrix (~array of doubles in [0, 1], 0 is white, 1 is
// black). The maximum such gray-scale value is returned.
double bmp_read(padded_matrix &image, const std::string& fname) {
    std::fstream fp;
    fp.open(fname, std::fstream::in | std::fstream::binary);
    if (fp.fail())
        die("cannot open the file " + fname);

    bmp_header header;

    fp.read((char *)(&header), sizeof(header));
    if (header.bi_bit_count != 24) {
        fp.close();
        die("not a 24-bit image in " + fname);
    }
    if (header.bi_compression) {
        fp.close();
        die(fname + " is compressed bmp");
    }
    if (header.bi_height <= 0 || header.bi_width <= 0) {
        fp.close();
        die("image " + fname + " has zero size");
    }
    if (header.bi_height > 65536 || header.bi_width > 65536) {
        fp.close();
        die("image " + fname + " is too large");
    }

    image.allocate(header.bi_height, header.bi_width);
    if (!image.data)
        die("no memory to read " + fname);

    fp.seekg(sizeof(header), std::ios_base::beg);
    pixel pix;
    double max_gray_scale = std::numeric_limits<double>::lowest();
    for (int i = 0; i < image.h; ++i) {
        for (int j = 0; j < image.w; ++j) {
            fp.read((char *)(&pix), 3);
            double gray_scale = (255.0 - (pix.r + pix.g + pix.b) / 3.0) / 255.0;
            image.data[i * image.ldw() + j] = gray_scale;
            max_gray_scale = std::max(max_gray_scale, gray_scale);
        }
        // rows are rounded up to a multiple of 4 bytes in BMP format
        fp.seekg((4 - (3 * image.w) % 4) % 4, std::ios_base::cur);
    }

    if (!fp)
        die("error reading " + fname);

    fp.close();
    return max_gray_scale;
}
// Routine exporting a padded_matrix (~array of doubles) as a gray-scale 24-bit
// uncompressed bitmap image file.
// For every entry v of the matrix, a gray scale (max(v, 0.0) / max_abs) is
// calculated and the corresponding pixel's color code is generated.
// Note: negative values are ignored (considered white) by this routine. While
// negative values may be found in the reconstructed image, they should be
// considered artifacts (e.g., due to Gibbs phenomenon) and/or local errors of
// the reconstruction procedure (e.g., due to the rudimentary spectrum
// interpolation).
void bmp_write(const std::string& fname, const padded_matrix &image) {
    unsigned sizeof_line  = (image.w * 3 + 3) / 4 * 4;
    unsigned sizeof_image = image.h * sizeof_line;

    bmp_header header = {{'B', 'M'},
                        unsigned(sizeof(header) + sizeof_image),
                        0,
                        sizeof(header),
                        sizeof(header) - offsetof(bmp_header, bi_size),
                        unsigned(image.w),
                        unsigned(image.h),
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
        die("failed to save the image, cannot open file " + fname);

    fp.write((char *)(&header), sizeof(header));
    double v_max = std::numeric_limits<double>::lowest();
    for (int i = 0; i < image.h; ++i)
        for (int j = 0; j < image.w; ++j)
            v_max = std::max(image.data[i * image.ldw() + j], v_max);

    if (v_max <= 0.0) {
        fp.close();
        die("inconsistent data range to consider for exporting " + fname);
    }
    pixel pix;
    for (int i = 0; i < image.h; ++i) {
        for (int j = 0; j < image.w; ++j) {
            const double gray =
                std::max(image.data[i * image.ldw() + j], 0.0)/v_max;
            pix.b = pix.g = pix.r =
                static_cast<unsigned char>(std::round(255.0*(1.0 - gray)));
            fp.write((char *)(&pix), 3);
        }
        // rows are rounded up to a multiple of 4 bytes in BMP format
        for (int j = 3 * image.w; j % 4; ++j)
            fp.put(0);
    }
    fp.close();
}

// For a given 2D function f(y, x), the radon transform of f is defined as
//               radon[f](theta, v) := integral of f along line L
// where L is the set of points (y, x) such that
//                    x*cos(theta) + y*sin(theta) = v
// in a given orthonormal frame.
// Let p = radon_projection.h, q = radon_projection.q, and
// radon_ldw = radon_projection.ldw(). For all integers i in [0, p-1] and j in
// [0, q-1], this routine sets
//    radon_projection.data[i*radon_ldw + j] = radon[f_input](theta(i), v(j))
// where
//   theta(i)   = -0.5*M_PI + i*M_PI/p,
//   v(j)       = (-1.0 + (2.0*j + 1.0)/q)*0.5*S,
// considering the function f_input defined as
//                      f_input(y, x) = 0
// if max(|y|/hh, |x|/ww) > 0.5, and
//            f_input(y, x) = input.data[ii*input.ldw() + jj]
// where integers ii and jj are such that
//              ii*in_pix_len <= y + 0.5*hh < (ii + 1)*in_pix_len,
//              jj*in_pix_len <= x + 0.5*ww < (jj + 1)*in_pix_len,
// using hh = input.h*in_pix_len and ww = input.w*in_pix_len.
//
// Inputs:
// -------
// S:                   scanning width used for sampling the Radon transform
//                      for every projection angle theta (see above);
// input:               padded_matrix defining the function f_input to consider
//                      when computing Radon transform values (see above).
// Output:
// -------
// radon_projection:    padded_matrix containing the desired samples of the
//                      radon transform described above.
//                      The calculations are enqueued to the queue encapsulated
//                      this object.
// Returns:
// --------
// a sycl::event object tracking the completion of the task.
sycl::event acquire_radon(padded_matrix &radon_projection,
                          double S, const padded_matrix &input) {

    const double hh = input.h*in_pix_len;
    const double ww = input.w*in_pix_len;
    if (S < std::hypot(hh, ww)) {
        die("invalid scanning width for acquire_radon");
    }
    // helper computing the length from "origin" to the closest point on one of
    // the lines that define a rectangular box (determined by its "center" and
    // its "side_lengths") in the direction of vector "+dir_v".
    // std::numeric_limits<double>::max() is returned if no intersection is
    // found.
    auto compute_length_to_closest_edge =
        [=](const double origin[2], const double dir_v[2],
            const double center[2], const double side_lengths[2]) {
        double lambda = std::numeric_limits<double>::max();
        const double abs_v = sycl::hypot(dir_v[0], dir_v[1]);
        for (int dim = 0; dim < 2; dim++) {
            if (dir_v[dim] == 0.0)
                continue;
            for (double bound : {center[dim] - 0.5*side_lengths[dim],
                                 center[dim] + 0.5*side_lengths[dim]}) {
                double candidate = abs_v * (bound - origin[dim]) / dir_v[dim];
                if (0.0 < candidate && candidate < lambda)
                    lambda = candidate;
            }
        }
        return lambda;
    };
    const int input_ldw = input.ldw();
    const double *input_data = input.data;
    auto integrate_along_line = [=](double theta, double v) {
        /*
        Let's consider an orthonormal reference frame centered at the center of
        the image (say O), using axes aligned with the edges of the image, as
        shown below.
        */
        double box_center[2] = {0.0, 0.0};
        double box_edges[2] = {hh, ww};
        /*
         line L (defined by theta and v)                            ^ (y-axis)
           \                                                        |
            \                    ww                                 |
            _\_______________________________________________       |
           |  \                                              |      |
           |   \                                             |      |
           |    \                                            |      |
           |     \                                           |      |
       hh  |      \               O                          |      - 0
           |       \                                         |      |
           |        \                                        |      |
           |         \                                       |      |
           |__________\______________________________________|      |
                       \                                            |
                        \         0                                 |
        --------------------------|------------------------------------>
        (x-axis)

        Using the following parameters,
        */
        const double cs = sycl::cos(theta), sn = sycl::sin(theta);
        const double unit_vector[2] = {+cs, -sn};
        const double half_chord = sycl::sqrt(0.25*S*S - v*v);
        /*
        the coordinates (y, x) of any point lying on the chord that L makes in
        the circle of center O and radius 0.5*S may be parameterized as
               y(alpha) = v*sn + (-half_chord + alpha)*unit_vector[0];
               x(alpha) = v*cs + (-half_chord + alpha)*unit_vector[1];
        where 0 <= alpha <= 2*half_chord is a curvilinear abscissa along L.
        The integral of the (piecewise constant) input signal for the range of
        alpha s.t. max(|y(alpha)/hh|, |x(alpha)/ww|) <= 0.5 is computed below
        */
        // cursor point on L (alpha = 0.0)
        double cursor[2] = {v * sn - half_chord * unit_vector[0],
                            v * cs - half_chord * unit_vector[1]};
        // Move the cursor along L (alpha += dalpha, dalpha > 0.0) into the
        // first pixel crossed by L. When moving the cursor, "a little more" is
        // added to dalpha to place it _into_ the first/next pixel of the image
        // crossed by L (avoiding ambiguous cases of pixel corners lying exactly
        // on L).
        constexpr double a_little_more = in_pix_len*1.0e-4;
        double dalpha = 0.0;
        while (sycl::fabs(cursor[0]) > 0.5*hh ||
               sycl::fabs(cursor[1]) > 0.5*ww) {
            dalpha = compute_length_to_closest_edge(cursor, unit_vector,
                                                    box_center, box_edges);
            if (dalpha > 2.0*half_chord) {
                // no intersection between L and the image was found
                return 0.0;
            }
            for (int dim = 0; dim < 2; dim++)
                cursor[dim] += (dalpha + a_little_more)*unit_vector[dim];
        }
        // parse the image pixels of indices (ii, jj) visited by the cursor as
        // alpha increases, i.e., as the cursor moves along L within the image
        int ii = static_cast<int>(sycl::floor((cursor[0] + 0.5*hh)/in_pix_len));
        int jj = static_cast<int>(sycl::floor((cursor[1] + 0.5*ww)/in_pix_len));
        // reset the box_edges to the size of individual pixels
        box_edges[0] = box_edges[1] = in_pix_len;
        // set box_center to the pixel's center
        box_center[0] = -0.5*hh + (ii + 0.5)*in_pix_len;
        box_center[1] = -0.5*ww + (jj + 0.5)*in_pix_len;
        // compute integral
        double integral = 0.0;
        while (sycl::fabs(box_center[0]) < 0.5*hh &&
               sycl::fabs(box_center[1]) < 0.5*ww) {
            // find smallest value of dalpha > 0 that brings the cursor to the
            // closest edge of the current pixel crossed by L.
            // There is always one dalpha s.t.
            //                0 < dalpha <= sqrt(2)*in_pix_len
            // for any pixel inside the input image, crossed by L.
            dalpha = compute_length_to_closest_edge(cursor, unit_vector,
                                                    box_center, box_edges);
            // Note: cursor was moved "a little more" into the pixel, previously
            integral += input_data[ii * input_ldw + jj]*(dalpha + a_little_more);
            // move the cursor into the following pixel
            for (int dim = 0; dim < 2; dim++)
                cursor[dim] += (dalpha + a_little_more)*unit_vector[dim];
            // set the new pixel indices
            ii = static_cast<int>(sycl::floor((cursor[0] + 0.5*hh)/in_pix_len));
            jj = static_cast<int>(sycl::floor((cursor[1] + 0.5*ww)/in_pix_len));
            box_center[0] = -0.5*hh + (ii + 0.5)*in_pix_len;
            box_center[1] = -0.5*ww + (jj + 0.5)*in_pix_len;
        }
        return integral;
    };

    sycl::event ev = radon_projection.q.submit([&](sycl::handler &cgh) {
        const int p = radon_projection.h;
        const int q = radon_projection.w;
        const int radon_ldw = radon_projection.ldw();
        double *radon_data  = radon_projection.data;
        cgh.parallel_for<class acquire_radon_class>(
            sycl::range<2>(p, q),
            [=](sycl::item<2> item) {
                const int i = item.get_id(0);
                const int j = item.get_id(1);
                // -0.5*M_PI <= theta < 0.5*M_PI
                const double theta = -0.5*M_PI + i * M_PI / p;
                // -0.5*S < v < 0.5*S
                const double v = (-1.0 + (2.0 * j + 1.0) / q)*0.5*S;
                radon_data[i * radon_ldw + j] = integrate_along_line(theta, v);
            }
        );
    });

    return ev;
}

// Routine reconstructing an S-by-S image of q x q pixels from p x q samples of
// its Radon transform (q samples spanning a scanning width S for every of the p
// projection directions).
//
// Inputs:
// -------
// R:     padded_matrix of height p and width q storing the samples of the radon
//        transform. Note that R is modified by this routine;
// S:     scanning width used when sampling the Radon transform;
// dep:   sycl::event object capturing the dependency to be honored before
//        accessing elements of R.
// Output:
// -------
// image: padded_matrix of height q and width q representing gray-scale pixel
//        values of the reconstructed image (square image of side length S).
void reconstruction_from_radon(padded_matrix& image,
                               padded_matrix& R, double S, sycl::event& dep) {
    if (S <= 0.0)
        die("invalid scanning width");
    const int p = R.h;
    const int q = R.w;
    image.allocate(q, q);
    if (!image.data)
        die("cannot allocate memory for reconstruction");
    std::cout << "Reconstructing image from the Radon projection data"
              << std::endl;
/*
    Note: in the explanatory comments below, arithmetic operations are to be
          understood as similar C++ instructions would, i.e., "x/2" represents
          the integer division of x by 2 if x is an integer itself, which is
          different from the value of 0.5*x if x%2 == 1.

    The input R to this routine is interpreted as
             R.data[i * R.ldw() + j] == radon[f](theta(i), v(j))
                                     := "R[i][j]"
    where
               theta(i) = -0.5*M_PI + i * M_PI / p;
               v(j)     = (-1.0 + (2.0*j + 1.0) / q) * 0.5 * S;
    for 0 <= i < p and 0 <= j < q.

    For a given value of theta, let radon_hat(theta, ksi) be the (continuous)
    1D Fourier transform of radon[f](theta, v), i.e., using "1i" := sqrt(-1),
        radon_hat(theta, ksi) :=
            integral (radon[f](theta, v)*exp(-1i*2*M_PI*v*ksi)) dv.
             v real
    By the Fourier slice theorem,
            radon_hat(theta, ksi) = f_hat(ksi*sin(theta), ksi*cos(theta))
    where f_hat is the (continuous) 2D Fourier transform of f, i.e.,
       f_hat(k_y, k_x) :=
           integral integral (f(y,x)*exp(-1i*2*M_PI*(k_y*y + k_x*x))) dx dy.
            y real   x real

    Considering ksi = r / S for integers r in [0, q/2], one has
        radon_hat(theta(i), r / S)
            =  integral (radon[f](theta(i), v)*exp(-1i*2*M_PI*(r/S)*v)) dv.
              |v| < S/2
    which may be approximated as (midpoint rule)
    radon_hat(theta(i), r / S)
        ~ \sum_j (R[i][j]*exp(-1i*2*M_PI*(r/S)*v(j))) * (S/q), for j=0,...,q-1
        = exp(1i*M_PI*r*(1.0 - 1.0/q)) * (S/q) * DFT(R[i][0:q[ ; r),    (eq. 1a)
    where DFT(R[i][0:q[; r) represents the r-th value of the forward DFT of
    the i-th row of q discrete values in R.
    Note: radon_hat(theta(i) + M_PI, r / S) = conj(radon_hat(theta(i), r / S))
*/
    std::cout << "\tStep 1 - Batch of " << p
              << " real 1D in-place forward DFTs of length " << q << std::endl;
    // Descriptor to compute the DFTs involved in the RHS of eq. 1a, scaled by
    // S/q
    real_descriptor_t radon_dft(q);
    // p values of theta
    radon_dft.set_value(dft_ns::config_param::NUMBER_OF_TRANSFORMS, p);
    // Distances must be set for batched transforms. For real in-place DFTs with
    // unit stride, the distance in forward domain (wherein elements are real)
    // must be twice the distance in backward domain (wherein elements are
    // complex). Therefore, padding is requied in forward domain (as accounted
    // for by the padded_matrix structure)
    radon_dft.set_value(dft_ns::config_param::FWD_DISTANCE, R.ldw());
    radon_dft.set_value(dft_ns::config_param::BWD_DISTANCE, R.ldw() / 2);
    // Scaling factor for forward DFT (see eq. 1a above)
    radon_dft.set_value(dft_ns::config_param::FORWARD_SCALE, S / q);
    // oneMKL DFT descriptor operate in-place by default
    radon_dft.commit(R.q);
    auto compute_radon_hat = dft_ns::compute_forward(radon_dft, R.data, {dep});
/*
    Using R_data_c = reinterpret_cast<complex_t*>(R.data), one has
         f_hat((r/S)*sin(theta(i)), (r/S)*cos(theta(i)))                (eq. 1b)
                 ~ exp(1i*M_PI*r*(1.0 - 1.0/q)) * R_data_c[i * R.ldw()/2 + r]
    for 0 <= i < p and 0 <= r <= q/2 (upon completion of compute_radon_hat).
    Note that values of i in [p, 2*p) are known too, as the complex conjugates
    of their (i-p) counterparts (albeit not stored explicitly).

    The reconstructed image is based off a version of the spectrum f_hat that
    is truncated to wave vectors no larger than 0.5*q/S in magnitude, e.g.,
      g_hat(ksi_y, ksi_x) =
              f_hat(ksi_y, ksi_x)*heaviside(0.5*q/S - sqrt(ksi_y^2 + ksi_x^2)),
    where heaviside(x) := {0, 0.5, 1} if {x < 0.0, x == 0, x > 0.0},
    respectively.

    While g(y, x) is not strictly equal to f(y, x) pointwise in general, g
    converges towards f in a weaker sense (e.g., in L2 norm) as q gets large if
    f is bounded and has compact support. Since samples of g are desired at
    specific points, i.e., the (centers of the) q x q pixels of the
    reconstructed S-by-S image, let
                  g_s(y, x) = g(y, x) * 2d_dirac_comb(y, x; S/q),
    where 2d_dirac_comb(y, x; alpha) is defined as
      \sum_j \sum_i 2d_dirac(y-alpha*j, x-alpha*i), {i,j} = -infty,...,+infty.

    By the convolution theorem, the spectrum of g_s is
        g_s_hat(ksi_y, ksi_x) =                                          (eq. 2)
     (q^2/S^2)*convolution[g_hat(., .), 2d_dirac_comb(., .; q/S)](ksi_y, ksi_x),
    which is periodic with period q/S along ksi_y and ksi_x. Sampling g_s_hat at
    wave vectors of components multiple of 1.0/S, i.e., considering
        f_tilde_hat(ksi_y, ksi_x) =                                      (eq. 3)
            g_s_hat(ksi_y, ksi_x) * 2d_dirac_comb(ksi_y, ksi_x; 1.0/S)
    effectively corresponds to f_tilde(y, x) being a periodic replication of
    (S^2)*g_s(y, x) with period S along y and x (assuming that values of
    g_s(y, x) may be ignored for {(y, x): max(|y|, |x|) > 0.5*S}).

    Developing the (continuous) 2D inverse Fourier transform of f_tilde_hat
    (see eq. 3), one has
      f_tilde(y, x) = \sum_i \sum_j 2d_dirac(y-i*(S/q), x-j*(S/q)) G[i%q][j%q],
                                                      {i,j} = -infty,...,+infty
    where G[i][j] (0 <= i < q. 0 <= j < q) is the (i,j) value of the 2D backward
    DFT of
           G_S_HAT[m][n] = g_s_hat(mm(m)/S, nn(n)/S), 0 <= m < q, 0 <= n < q
    where 0 <= |xx(x)| <= q/2 and mod(x, q) == mod(xx(x), q) with 'x' being
    either 'm' or 'n'. Explicitly, for 0 <= m < q, 0 <= n < q, let
        abs_S_ksi   = sycl::hypot(mm(n), nn(n));
        theta       = sycl::atan2(mm(n), nn(n));
    then (see eq. 2)
        G_S_HAT[m][n]=0.0,                                 if abs_S_ksi > 0.5*q;
        G_S_HAT[m][n]=f_hat(mm(m)/S, nn(n)/S),      if round(abs_S_ksi) < 0.5*q;
        G_S_HAT[m][n]=0.5*f_hat(mm(m)/S, nn(n)/S),  if round(abs_S_ksi) = 0.5*q,
                                               and fmod(theta, 0.5*M_PI) != 0.0;
        G_S_HAT[m][n]=real_part(f_hat(mm(m)/S, nn(n)/S)),
                                                    if round(abs_S_ksi) = 0.5*q,
                                                and fmod(theta, 0.5*M_PI) = 0.0.

    Identifying f_tilde(y, x) as the periodic replication of (S^2)*g_s(y, x)
    with period S along y and x, one concludes that
                 g(i*(S/q), j*(S/q)) = G[i%q][j%q] / S^2                 (eq. 4)
    for integers i and j s.t. |i| <= q/2 and |j| <= q/2 (assuming that values of
    g_s(y, x) may be ignored for {(y, x): max(|y|, |x|) > 0.5*S}).

    The required values of G_S_HAT[m][n] are interpolated below from the known
    values stored in R (using eq. 1b).
    Note that G_S_HAT[q - m][q - n] = conj(G_S_HAT[m][n]) given that
    f_hat(-ksi_y, -ksi_x) = conj(f_hat(ksi_y, ksi_x)). This is consistent with
    the requirements for a well-defined backward real 2D DFT, and the values of
    G_S_HAT[m][n] do not need to be set explicitly for n > q/2.
*/
    std::cout << "\tStep 2 - Interpolating spectrum from polar to "
              << "cartesian grid" << std::endl;
    auto interp_ev = image.q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(compute_radon_hat);
        const complex_t *R_data_c = reinterpret_cast<complex_t*>(R.data);
        complex_t *G_S_HAT = reinterpret_cast<complex_t*>(image.data);
        const int R_data_c_ldw  = R.ldw()/2;        // in complex values
        const int G_S_HAT_ldw   = image.ldw()/2;    // in complex values
        cgh.parallel_for<class interpolateKernelClass>(
            sycl::range<2>(q, q/2 + 1),
            [=](sycl::item<2> item) {
                const int m = item.get_id(0);
                const int n = item.get_id(1);
                const int mm = m - (2*m > q ? q : 0);
                const int nn = n; // nn(n) == n since n never exceeds q/2
                const double abs_S_ksi = sycl::hypot(double(mm), double(nn));
                const double theta = (mm == 0 && nn == 0) ?
                        0.0 : sycl::atan2(double(mm), double(nn));
                // theta in [-0.5*M_PI, +0.5*M_PI]
                complex_t G_S_HAT_mn = complex_t(0.0, 0.0);
                if (abs_S_ksi <= 0.5*q) {
                    const int r = static_cast<int>(sycl::round(abs_S_ksi));
                    const int i =
                        static_cast<int>(sycl::round(((theta + 0.5*M_PI)/M_PI)*p));
                    // if i < 0 or i >= p (e.g., theta = 0.5*M_PI corresponds to
                    // i == p), the value is mapped to the complex conjugate of
                    // R_data_c[(i%p) * R_data_c_ldw + r] (e.g.,
                    // theta = -0.5*M_PI if i == p).
                    complex_t f_hat_mm_nn =
                        R_data_c[(i% p) * R_data_c_ldw + r]*
                        complex_t(sycl::cos(M_PI*r*(1.0 - 1.0/q)),
                                  sycl::sin(M_PI*r*(1.0 - 1.0/q))); // see eq. 1b
                    if (i%(2*p) >= p)
                        f_hat_mm_nn = std::conj(f_hat_mm_nn);

                    G_S_HAT_mn = f_hat_mm_nn;
                    if (2*r == q) {
                        if (nn == 0 || mm == 0) // |theta| = 0.0 or 0.5*M_PI
                            G_S_HAT_mn.imag(0.0);
                        else
                            G_S_HAT_mn *= 0.5;
                    }
                    // For a more convenient representation of the reconstructed
                    // image, shift the target reference frame so that the
                    // center of the reconstructed image is located at pixel
                    // of indices (q/2, q/2):
                    G_S_HAT_mn *= complex_t(sycl::cos(-2.0*M_PI*m*(q/2)/q),
                                            sycl::sin(-2.0*M_PI*m*(q/2)/q));
                    // RHS is ((-1)^m, 0) is q is even
                    G_S_HAT_mn *= complex_t(sycl::cos(-2.0*M_PI*n*(q/2)/q),
                                            sycl::sin(-2.0*M_PI*n*(q/2)/q));
                    // RHS is ((-1)^n, 0) is q is even
                }
                G_S_HAT[m * G_S_HAT_ldw + n] = G_S_HAT_mn;
        });
    });
    std::cout << "\tStep 3 - In-place backward real 2D DFT of size "
              << q << "x" << q << std::endl;
    real_descriptor_t q_by_q_real_dft({q, q});
    // Default strides are set by default for in-place DFTs (consistently with
    // the implementation of padded_matrix)
    // Scaling factor for backward DFT (see eq. 4)
    q_by_q_real_dft.set_value(dft_ns::config_param::BACKWARD_SCALE, 1.0 / (S*S));
    q_by_q_real_dft.commit(image.q);
    auto compute_g_values =
        dft_ns::compute_backward(q_by_q_real_dft, image.data, {interp_ev});
    compute_g_values.wait();
}

// Routine computing the mean global error and pixel-wise mean local errors in
// the reconstructed image, when compared to the (gray-scale-converted)
// original image. The mean error over an area A is defined as
//       (1.0/area(A)) integral |f_reconstruction - f_original| dA
//                        A
// where f_reconstruction (resp. f_original) is the piecewise contant function
// of the gray-scale intensity in the reconstructed (resp. original) image.
// Note: f_reconstruction and f_original are considered equal to 0.0 outside
// the support of the original image.
//
// Inputs:
// -------
// S:               scanning width used when sampling the Radon transform;
// original:        padded_matrix containing the original input image's pixel
//                  values converted to gray-scale values;
// reconstruction:  padded_matrix containing the reconstructed image's pixel
//                  gray-scale values.
// Output:
// -------
// errors:          padded_matrix of the same size as reconstruction, containing
//                  the pixel-wise mean local errors in the pixels of the
//                  reconstructed image.
// Returns:
// --------
// The mean global error.
double compute_errors(padded_matrix& errors,
                      double S,
                      const padded_matrix& original,
                      const padded_matrix& reconstruction) {
    const int q = reconstruction.h;
    if (q <= 0 || q != reconstruction.w)
        die("invalid reconstruction considered for evaluating mean local errors");
    errors.allocate(q, q);
    if (!errors.data)
        die("cannot allocate memory for mean local errors");
    // the reconstructed image is an S-by-S image of q x q pixels
    const double pixel_length = S/q;
    const double max_overlap = sycl::min(pixel_length, in_pix_len);
    // dimensions of the original image:
    const double hh = in_pix_len*original.h;
    const double ww = in_pix_len*original.w;

    // helper routines to compute the mean local error in pixel (i,j) of the
    // reconstructed image
    const int supremum_io = original.h - 1;
    const int supremum_jo = original.w - 1;
    auto get_original_index = [=](double y, double x) {
        int io = static_cast<int>(sycl::floor((0.5*hh + y) / in_pix_len));
        int jo = static_cast<int>(sycl::floor((0.5*ww + x) / in_pix_len));
        io = sycl::max(0, sycl::min(io, supremum_io));
        jo = sycl::max(0, sycl::min(jo, supremum_jo));
        return std::pair<int, int>(io, jo);
    };
    const double* reconstruction_data = reconstruction.data;
    const double* original_data = original.data;
    const int reconstruction_ldw = reconstruction.ldw();
    const int original_ldw = original.ldw();
    auto compute_mean_error_in_pixel = [=](int i, int j) {
        // Notes:
        // - pixel of index (i,j) in the reconstructed image covers the area of
        //   points (y, x) such that
        //              |y - (i - q/2)*pixel_length| <= 0.5*pixel_length
        //              |x - (j - q/2)*pixel_length| <= 0.5*pixel_length
        // - pixel of index (io,jo) in the original image covers the area of
        //   points (y, x) such that
        //          io*in_pix_len <= y + 0.5*hh <= (io+1)*in_pix_len
        //          jo*in_pix_len <= x + 0.5*ww <= (jo+1)*in_pix_len
        // in a common orthonormal reference frame centered on either image
        const double y_min = (i - q/2 - 0.5)*pixel_length;
        const double x_min = (j - q/2 - 0.5)*pixel_length;
        const double y_max = (i - q/2 + 0.5)*pixel_length;
        const double x_max = (j - q/2 + 0.5)*pixel_length;
        if (y_min > 0.5*hh || y_max < -0.5*hh ||
            x_min > 0.5*ww || x_max < -0.5*ww) {
            // out of scope of relevance, no intersection with the original
            // image
            return 0.0;
        }
        // find corresponding range of pixels in original image
        const auto io_jo_min = get_original_index(y_min, x_min);
        const auto io_jo_max = get_original_index(y_max, x_max);
        const double got = reconstruction_data[i*reconstruction_ldw + j];
        double mean_local_error = 0.0;
        // parse original image's pixels (io, jo) having a non-empty
        // intersection with the reconstructed image's pixel (i, j)
        for (int io = io_jo_min.first; io <= io_jo_max.first; io++) {
            const double yo_min = -0.5*hh + io*in_pix_len;
            const double yo_max = -0.5*hh + (io + 1)*in_pix_len;
            double overlap_y =
                sycl::min(y_max, yo_max) - sycl::max(y_min, yo_min);
            overlap_y = sycl::max(0.0, sycl::min(overlap_y, max_overlap));
            for (int jo = io_jo_min.second; jo <= io_jo_max.second; jo++) {
                const double xo_min = -0.5*ww + jo*in_pix_len;
                const double xo_max = -0.5*ww + (jo + 1)*in_pix_len;
                double overlap_x =
                    sycl::min(x_max, xo_max) - sycl::max(x_min, xo_min);
                overlap_x = sycl::max(0.0, sycl::min(overlap_x, max_overlap));
                const double exp = original_data[io*original_ldw + jo];
                mean_local_error +=
                    sycl::fabs(exp - got)*overlap_y*overlap_x;
            }
        }
        mean_local_error /= (pixel_length*pixel_length);
        return mean_local_error;
    };
    // compute the mean local errors and the mean global error
    double mean_error = 0.0;
    double* L1_error = (double *)sycl::malloc_device(sizeof(double), errors.q);
    if (!L1_error)
        die("cannot allocate memory for mean global error");
    auto init_ev = errors.q.copy(&mean_error, L1_error, 1);
    auto compute_errors_ev = errors.q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(init_ev);
        auto global_integral = sycl::reduction(L1_error, sycl::plus<>());
        double *local_error_data = errors.data;
        const int local_error_ldw = errors.ldw();
        cgh.parallel_for<class compute_errors_class>(
            sycl::range<2>(q, q), global_integral,
            [=](sycl::item<2> item, auto& sum) {
                const int i = item.get_id(0);
                const int j = item.get_id(1);
                const double mean_err_ij = compute_mean_error_in_pixel(i, j);
                local_error_data[i*local_error_ldw + j] = mean_err_ij;
                sum += mean_err_ij * pixel_length * pixel_length;
            }
        );
    });
    errors.q.copy(L1_error, &mean_error, 1, compute_errors_ev).wait();
    sycl::free(L1_error, errors.q);
    mean_error /= (hh*ww);
    return mean_error;
}

int main(int argc, char **argv) {
    const int p                         = argc > 1 ? std::atoi(argv[1]) :
                                                     default_p;
    const int q                         = argc > 2 ? std::atoi(argv[2]) :
                                                     default_q;
    const double S_to_D                 = argc > 3 ? std::atof(argv[3]) :
                                                     default_S_to_D;
    const std::string original_bmpname  = argc > 4 ? argv[4] :
                                                     default_original_bmpname;
    const std::string radon_bmpname     = argc > 5 ? argv[5] :
                                                     default_radon_bmpname;
    const std::string restored_bmpname  = argc > 6 ? argv[6] :
                                                     default_restored_bmpname;
    const std::string errors_bmpname    = argc > 7 ? argv[7] :
                                                     default_errors_bmpname;
    // validate numerical input arguments
    if (argc > 8 || p <= 0 || q <= 0 || S_to_D < 1.0) {
        die("invalid usage.\n" + usage_info);
    }

    // Create execution queue.
    sycl::queue main_queue;
    try {
        // This sample requires double-precision floating-point arithmetic:
        main_queue = sycl::queue(sycl::aspect_selector({sycl::aspect::fp64}));
    } catch (sycl::exception &e) {
        std::cerr << "Could not find any device with double precision support."
                  << "Exiting." << std::endl;
        return 0;
    }
    // read input image and convert it to gray-scale values
    std::cout << "Reading original image from " << original_bmpname << std::endl;
    padded_matrix original(main_queue);
    const double max_input_value = bmp_read(original, original_bmpname);
    // diagonal D of original image
    const double D = std::hypot(original.h, original.w)*in_pix_len;
    // scanning width S
    const double S = S_to_D * D;
    // Compute samples of the radon transform
    padded_matrix radon_image(main_queue);
    radon_image.allocate(p, q);
    if (!radon_image.data)
        die("cannot allocate memory for Radon projection");
    std::cout << "Generating Radon transform data from "
              << original_bmpname << std::endl;
    auto radon_ev = acquire_radon(radon_image, S, original);
    std::cout << "Saving Radon transform data in "
              << radon_bmpname << std::endl;
    radon_ev.wait(); // make sure it completes before exporting data
    bmp_write(radon_bmpname, radon_image);
    // reconstruct image from its radon transform samples
    padded_matrix reconstruction(main_queue);
    reconstruction_from_radon(reconstruction, radon_image, S, radon_ev);
    std::cout << "Saving restored image in " << restored_bmpname << std::endl;
    bmp_write(restored_bmpname, reconstruction);
    // evaluate the mean error, pixel by pixel in the reconstructed image
    padded_matrix errors(main_queue);
    const double mean_error = compute_errors(errors, S, original, reconstruction);
    std::cout << "The mean error in the reconstruction image is " << mean_error
              << std::endl;
    if (mean_error / max_input_value > arbitrary_error_threshold) {
        std::cerr << "The mean error exceeds the (arbitrarily-chosen) "
                  << "threshold of " << 100.0*arbitrary_error_threshold
                  << "% of the original image's maximum gray-scale value"
                  << std::endl;
        if (std::fabs(p * S_to_D - q) > 0.2*std::max(p*S_to_D, double(q))) {
            std::cerr << "It is recommended to use values of p and q such that "
                      << "p*S_to_D and q are commensurate." << std::endl;
        }
        else if (S / q > 2.0*in_pix_len) {
            std::cerr << "Consider increasing q (to "
                      << std::ceil(S/(2.0*in_pix_len)) << " or more) "
                      << "to alleviate blurring in the reconstructed image."
                      << std::endl;
        }
        else {
            std::cerr << "Consider increasing S_to_D and q proportionally to "
                      << "one another to reduce interpolation errors."
                      << std::endl;
        }
        std::cerr << "Saving local errors in " << errors_bmpname
                  << std::endl << std::flush;
        bmp_write(errors_bmpname, errors);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
