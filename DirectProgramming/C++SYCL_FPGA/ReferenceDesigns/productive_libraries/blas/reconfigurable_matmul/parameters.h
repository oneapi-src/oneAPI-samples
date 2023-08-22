#ifndef MATMUL_CONST_PARAMS_H
#define MATMUL_CONST_PARAMS_H

// NOTE: When changing any parameters of the systolic array (KKK, JJJ, III, JJ, II, KK), change function get_systolic_array_dimensions() in api.hpp accordingly

#ifdef TINY // For verifying correctness only
    #define KKK         4
    #define JJJ         4
    #define III         4
    #define JJ          4
    #define II          4
    #define KK          4
#else // LARGE
    #if defined(S10)
        #ifdef TYPEC_S
            #define KKK         16
            #define JJJ         16
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_D
            #define KKK         8
            #define JJJ         4
            #define III         6
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_C
            #define KKK         8
            #define JJJ         4
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_Z
            #define KKK         4
            #define JJJ         6
            #define III         3
            #define JJ          32
            #define II          32
            #define KK          32
        #else
            #error Precision of the output matrix is undefined.
        #endif
    #elif defined(A10)
        #ifdef TYPEC_S
            #define KKK         16
            #define JJJ         8
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_D
            #define KKK         8
            #define JJJ         4
            #define III         6
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_C
            #define KKK         8
            #define JJJ         4
            #define III         10
            #define JJ          32
            #define II          32
            #define KK          32
        #elif TYPEC_Z
            #define KKK         4
            #define JJJ         4
            #define III         3
            #define JJ          32
            #define II          32
            #define KK          32
        #else
            #error Precision of the output matrix is undefined.
        #endif
    #else
        #error The sizes of the systolic array are undefined.
    #endif
#endif

#if TYPEC_S
    #define ZERO        0
    #define SCALAR_ZERO 0
#elif TYPEC_D
    #define ZERO       0
#elif TYPEC_C
    #define ZERO       complex32_t(0.0f, 0.0f)
#elif TYPEC_Z
    #define ZERO       complex64_t(0.0, 0.0)
#else
    #error Precision of the output matrix is undefined.
#endif

#if TYPE_SCALAR_S
    #define SCALAR_ZERO 0
#elif TYPE_SCALAR_D
    #define SCALAR_ZERO 0
#elif TYPE_SCALAR_C
    #define SCALAR_ZERO complex32_t(0.0f, 0.0f)
#elif TYPE_SCALAR_Z
    #define SCALAR_ZERO complex64_t(0.0, 0.0)
#else
    #error Precision of beta and alpha is undefined.
#endif

#endif
