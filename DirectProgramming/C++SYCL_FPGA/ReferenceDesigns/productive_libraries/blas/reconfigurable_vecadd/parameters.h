#pragma once

#ifdef TINY
    #define KK 4
#else
    #if defined(S10)
        #ifdef T2SP_SVECADD
            #define KK 16
        #elif defined(T2SP_DVECADD)
            #define KK 8
        #elif defined(T2SP_CVECADD)
            #define KK 8
        #elif defined(T2SP_ZVECADD)
            #define KK 4
        #endif
    #elif defined(A10) // A10
        #ifdef T2SP_SVECADD
            #define KK 16
        #elif defined(T2SP_DVECADD)
            #define KK 8
        #elif defined(T2SP_CVECADD)
            #define KK 8
        #elif defined(T2SP_ZVECADD)
            #define KK 4
        #endif
    #else
        #error No FPGA hardware platform (A10 or S10) specified
    #endif
#endif

#ifdef T2SP_SVECADD
    #define TTYPE Float(32)
    #define CONST_TYPE float
#elif defined(T2SP_DVECADD)
    #define TTYPE Float(64)
    #define CONST_TYPE double
#elif defined(T2SP_CVECADD)
    #define TTYPE Complex(32)
    #define CONST_TYPE complex32_t
#elif defined(T2SP_ZVECADD)
    #define TTYPE Complex(64)
    #define CONST_TYPE complex64_t
#endif
