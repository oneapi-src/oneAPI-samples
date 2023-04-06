/**
   N - number of instances, power of 2, range = [1, 2^20]
   count - number of T types per instance, range = [1, 2^20/(sizeof(T)*8)]
   b - small modulos base size, range = [1, 2^14/(sizeof(T)*8)], RNS math, b = 0 - no base, generic modulo op
   T - 32, 64, 128bits
*/

// BigInt ops
template <int N, typename T>
void XeHE_add(
            const T *operand1,
			size_t operand1_count,
			const T *operand2,
            size_t operand2_count,
			size_t result_count,
			T *result);
			
template <int N, typename T>
void XeHE_sub(
            const T *operand1,
			size_t operand1_count,
			const T *operand2,
            size_t operand2_count,
			size_t result_count,
			T *result);
			
template <int N, typename T>
void XeHE_mul(
            const T *operand1,
			size_t operand1_count,
			const T *operand2,
            size_t operand2_count,
			size_t result_count,
			T *result);

template <int N, typename T>
void XeHE_mad(
            const T *operand1,
			size_t operand1_count,
			const T *operand2,
            size_t operand2_count,
			const T *operand3,
            size_t operand3_count,            
			size_t result_count,
			T *result);            
	
template <int N, typename T>	
void XeHE_right_shift(
            T *operand,
			int shift_amount,
			size_t operand_count,
			T *result);	
			
template <int N, typename T>	
void XeHE_left_shift(
            T *operand,
			int shift_amount,
			size_t operand_count,
			T *result);
			 
template <int N, typename T>
void XeHE_div_inplace(
			T *dividend_numerator,
			size_t divident_count,
			T *diviser,
			size_t diviser_count,
			T *quotient);	

// Mod ops			
template <int N, int b, typename T>
T XeHE_modulo(
            const T *value,
			size_t value_count,
			const T *modulus,
            const T *inv_modulus,
            size_t modulus_count);			

template <int N, int b, typename T>
T XeHE_barrett_reduce(
            const T *input,
            size_t input_count,
            const T *modulus,
            const T *inv_modulus,
            size_t modulus_count);
	   
template <int N, int b, typename T> 
void XeHE_add_mod(
            const T * operand1,
            const T * operand2,
            size_t operand_count,
            const T *modulus,
            size_t modulus_count,
            T * result);

template <int N, int b, typename T> 
T XeHE_mul_mod(
            const T * operand1,
            const T * operand2,
            size_t operand_count,
            const T *modulus,
            const T *inv_modulus,
            size_t modulus_count,
            T * result);

template <int N, int b, typename T> 
T XeHE_mad_mod(
            const T * operand1,
            const T * operand2,
            const T * operand3,
            size_t operand_count,
            const T *modulus,
            const T *inv_modulus,
            size_t modulus_count,
            T * result);
		
