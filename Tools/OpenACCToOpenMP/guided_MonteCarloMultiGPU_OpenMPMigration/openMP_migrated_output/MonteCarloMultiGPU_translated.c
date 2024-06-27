#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#define A1 0.31938153
#define A2 -0.356563782
#define A3 1.781477937
#define A4 -1.821255978
#define A5 1.330274429
#define RSQRT2PI 0.39894228040143267793994605993438
#define OPT_N 256
#define ABS(A) ((A) > 0 ? (A) : -(A))
typedef struct{
    float S;
    float X;
    float T;
    float R;
    float V;
} TOptiondata;


float rand_float(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

#pragma omp declare target
double MoroInvCND(double P){
    const double a1 = 2.50662823884;
    const double a2 = -18.61500062529;
    const double a3 = 41.39119773534;
    const double a4 = -25.44106049637;
    const double b1 = -8.4735109309;
    const double b2 = 23.08336743743;
    const double b3 = -21.06224101826;
    const double b4 = 3.13082909833;
    const double c1 = 0.337475482272615;
    const double c2 = 0.976169019091719;
    const double c3 = 0.160797971491821;
    const double c4 = 2.76438810333863E-02;
    const double c5 = 3.8405729373609E-03;
    const double c6 = 3.951896511919E-04;
    const double c7 = 3.21767881768E-05;
    const double c8 = 2.888167364E-07;
    const double c9 = 3.960315187E-07;
    double y, z;

    y = P - 0.5;
    if(fabs(y) < 0.42){
        z = y * y;
        z = y * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1);
    }else{
        if(y > 0)
            z = log(-log(1.0 - P));
        else
            z = log(-log(P));

        z = c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)))))));
        if(y < 0) z = -z;
    }

    return z;
}
#pragma omp end declare target

#pragma omp declare target
double NormalDistribution(unsigned int i, unsigned int pathN){
    double p = (double)(i + 1) / (double)(pathN + 1);
    return MoroInvCND(p);
}
#pragma omp end declare target

#pragma omp declare target
static double endCallValue(double S, double X, double r, double MuByT, double VBySqrtT){
    double callValue = S * exp(MuByT + VBySqrtT * r) - X;
    return (callValue > 0) ? callValue : 0;
}
#pragma omp end declare target

void MonteCarlo(float *call_value_e, float *confidence, TOptiondata option_data,unsigned int path_n)
{
    const double S = option_data.S;
    const double X = option_data.X;
    const double T = option_data.T;
    const double R = option_data.R;
    const double V = option_data.V;
    const double mu_x_t = (R - 0.5 * V * V) * T;
    const double v_x_sqrt_t = V * sqrtf(T);

    double sum = 0, sum2 = 0;

    for(int pos = 0; pos < path_n; pos++) {
        double sample = NormalDistribution(pos, path_n);
        double call_value = endCallValue(S, X, sample, mu_x_t, v_x_sqrt_t);
        sum  += call_value;
        sum2 += call_value * call_value;
    }

    *call_value_e = (float)(expf(-R * T) * sum / (double)path_n);
    double stdDev = sqrtf(((double)path_n * sum2 - sum * sum)/ ((double)path_n * (double)(path_n - 1)));
    *confidence = (float)(expf(-R * T) * 1.96 * stdDev / sqrtf((double)path_n));
}

void MonteCarloCPU(float *call_value_e_cpu, float *confidence_cpu, TOptiondata *option_data_arr, int path_n)
{
   double start = omp_get_wtime();

    for(int i = 0; i < OPT_N; i++){
        MonteCarlo(call_value_e_cpu + i, confidence_cpu + i, option_data_arr[i], path_n);
    }
    double end = omp_get_wtime();
    double accMs = end- start;

    printf("MonteCarloCPU() used time: %f (ms)\n", accMs);
}

void MonteCarloMultiGPU(float *call_value_e_gpu, float *confidence_gpu, TOptiondata *option_data_arr, int path_n)
{
    int gpu_n = omp_get_num_devices();
    double start;

#pragma omp target enter data map(to:option_data_arr[0:OPT_N])\
            map(alloc:call_value_e_gpu[0:OPT_N],confidence_gpu[0:OPT_N])
    {
   start = omp_get_wtime();    
    int len = OPT_N / gpu_n;
    int rem = OPT_N % gpu_n;
    int s, e;
    for (int j = 0; j < gpu_n; j++) {
        // set start point and end point for every part
        if (j < rem) {
            s = j * (len + 1);
            e = s + len;
        } else {
            s = j * len + rem;
            e = s + len - 1;
        }

#pragma omp target teams loop map(present,alloc:call_value_e_gpu[0:OPT_N],\
            confidence_gpu[0:OPT_N],option_data_arr[0:OPT_N]) nowait
       for (int i = s; i <= e; i++) {
            float *call_value_e = call_value_e_gpu + i, *confidence = confidence_gpu + i;
            TOptiondata option_data = option_data_arr[i];

            float S = option_data.S;
            float X = option_data.X;
            float T = option_data.T;
            float R = option_data.R;
            float V = option_data.V;
            float mu_x_t = (R - 0.5f * V * V) * T;
            float v_x_sqrt_t = V * sqrtf(T);

            float sum = 0, sum2 = 0;

#pragma omp loop reduction(+:sum,sum2)
            for(unsigned int pos = 0; pos < path_n; ++pos) {
                float sample = NormalDistribution(pos, path_n);
                float call_value = endCallValue(S, X, sample, mu_x_t, v_x_sqrt_t);
                sum  += call_value;
                sum2 += call_value * call_value;
            }

            *call_value_e = (float)(expf(-R * T) * sum / (float)path_n);
            float stdDev = sqrtf(((float)path_n * sum2 - sum * sum)/ ((float)path_n * (float)(path_n - 1)));
            *confidence = (float)(expf(-R * T) * 1.96f * stdDev / sqrtf((float)path_n));
        }
    }
    }
    #pragma omp taskwait
    double end = omp_get_wtime();
    double accMs = end - start;
    printf("MonteCarloMultiGPU() used time: %f (ms)\n", accMs);
#pragma omp target exit data map(from:call_value_e_gpu[0:OPT_N],\
            confidence_gpu[0:OPT_N]) map(delete:option_data_arr[0:OPT_N])

}
int32_t fcheck(float *A, float *B, uint32_t N, float th)
{
    int i;
    for (i = 0; i < N; i++) {
        if (ABS(A[i] - B[i]) > th) {
            printf("Test %d out of %d FAILED, %f %f\n", i, N, A[i], B[i]);
            return 1;
        }
    }
    return 0;
}

void runtest(float thresh)
{
    unsigned int time(void *);
    float call_value_e_gpu[OPT_N], confidence_gpu[OPT_N];
    float call_value_e_cpu[OPT_N], confidence_cpu[OPT_N];
    TOptiondata option_data[OPT_N];
    int path_n = 1 << 18, i;

    int GPU_N = omp_get_num_devices();
    printf("Number of GPUs          = %d\n", GPU_N);

    printf("main(): generating input data...\n");
    // init with random data
    srand(time(NULL));
    for(i = 0; i < OPT_N; ++i){
        option_data[i].S = rand_float(5.0f, 50.0f);
        option_data[i].X = rand_float(10.0f, 25.0f);
        option_data[i].T = rand_float(1.0f, 5.0f);
        option_data[i].R = 0.06f;
        option_data[i].V = 0.10f;
        call_value_e_gpu[i]  = -1.0f;
        call_value_e_cpu[i]  = -1.0f;
        confidence_gpu[i] = -1.0f;
        confidence_cpu[i] = -1.0f;
    }

    printf("running CPU MonteCarlo...\n");
    MonteCarloCPU(call_value_e_cpu, confidence_cpu, option_data, path_n);
#if defined(OPENACC2OPENMP_ORIGINAL_OPENMP)
    #pragma omp parallel num_threads(GPU_N)
#endif // defined(OPENACC2OPENMP_ORIGINAL_OPENMP)
    {
        omp_set_default_device(omp_get_thread_num());
        printf("GPU Device #%d\n", omp_get_default_device());
        MonteCarloMultiGPU(call_value_e_gpu, confidence_gpu, option_data, path_n);
        printf("%s\n", (fcheck(call_value_e_cpu, call_value_e_gpu, OPT_N, thresh) ? "Test FAILS" : "Test PASSES"));
    }

    printf("Number of options: %d\nNumber of paths: %d\n", OPT_N, path_n);
}

int main(int argc, char **argv)
{
    float th = 0.1;

    printf("%s Starting...\n\n", argv[0]);
    printf("MonteCarloMultiGPU\n");
    printf("==================\n");

    runtest(th);

    return 0;
}
