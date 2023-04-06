/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch2/catch.hpp"



#ifdef BUILD_WITH_HEAAN
#include <NTL/BasicThreadPool.h>
#include "Ciphertext.h"
#include "EvaluatorUtils.h"
#include "Ring.h"
#include "Scheme.h"
#include "SchemeAlgo.h"
#include "SecretKey.h"
#include "StringUtils.h"
#include "TimeUtils.h"

TEST_CASE("Basic test with HEAAN", "[example][cpu][HEAAN]") {
    long logq = 1200; ///< Ciphertext Modulus
    long logp = 30; ///< Real message will be quantized by multiplying 2^40
    long logn = 5; ///< log2(The number of slots)

    srand(time(NULL));
    SetNumThreads(8);
    TimeUtils timeutils;
    Ring ring;
    SecretKey secretKey(ring);
    Scheme scheme(secretKey, ring);

    long size = (1 << logn);
    complex<double> *mvec = EvaluatorUtils::randomComplexArray(size);
    Ciphertext cipher;

    timeutils.start("Encrypt");
    scheme.encrypt(cipher, mvec, size, logp, logq);
    timeutils.stop("Encrypt");

    timeutils.start("Decrypt");
    complex<double> *dvec = scheme.decrypt(secretKey, cipher);
    timeutils.stop("Decrypt");

    string prefix = "val";
    //StringUtils::compare(mvec, dvec, size, prefix);
    double epsilon = 1.0e-7;
    bool verbose_flag = false;
    for (long i = 0; i < size; ++i) {
        if (verbose_flag) {
            cout << "---------------------" << endl;
            cout << "m" + prefix + ": " << i << " :" << mvec[i] << endl;
            cout << "d" + prefix + ": " << i << " :" << dvec[i] << endl;
            cout << "e" + prefix + ": " << i << " :" << (mvec[i]-dvec[i]) << endl;
            cout << "---------------------" << endl;
        }
        complex<double> diff = mvec[i] - dvec[i];
        REQUIRE(diff.imag() < epsilon);
        REQUIRE(diff.real() < epsilon);
    }
}

#endif

TEST_CASE("XeHE basic test with HEAAN", "[example][cpu]") {

    int x = 2;
    SECTION("mult"){
        REQUIRE(x*3 == 6);
    }
    SECTION("square"){
        REQUIRE(x*x == 4);
    }
}
