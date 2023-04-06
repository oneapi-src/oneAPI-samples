/**
   T - 32, 64, 128bits
   all inputs < modulus
   inv_modulus - (2^log(sizeof(T)) / modulus)
*/

// Mod ops			
//  value % modulus -> return 		
template <typename T>
T XeHE_barrett_reduce(
            const T value,
            const T modulus,
            const T inv_modulus[2]
            );

// 	(operand1 + operand2) % modulus -> return   
template <typename T> 
T XeHE_add_mod(
            const T operand1,
            const T operand2,
            const T modulus);

// 	(operand1 - operand2) % modulus -> return   
template <typename T> 
T XeHE_sub_mod(
            const T operand1,
            const T operand2,
            const T modulus);

// 	(operand1 * operand2) % modulus -> return   
template <typename T> 
T XeHE_mul_mod(
            const T operand1,
            const T operand2,
            const T modulus,
            const T inv_modulus[2]);

// 	(operand1 * operand2 + operand3) % modulus -> return   
template <typename T> 
T XeHE_mad_mod(
            const T operand1,
            const T operand2,
            const T operand3,
            const T modulus,
            const T inv_modulus[2]);
		
