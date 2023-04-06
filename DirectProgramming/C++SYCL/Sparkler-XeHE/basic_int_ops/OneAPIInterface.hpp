#pragma once
Utils
GetPrimes
GetCoprimes
IsPrime
IsCoprime

Basic BigINT OPs

AddWithCarry<T>
SubWithBorrow<T>
Mul<T>                   TxT -> 2T
R / LShift<T>            shift <= T
R / LShift2<T>          shift <= T * 2
R / LShift3<T>          shift <= T * 3
Rotate
Divide2 < T>             2T / T->quotient<T>, reminder<T>
Divide3 < T>             3T / T->quotient<2T>, reminder<T>

Mod Ops, generic operand(s); operand(s) % T
AddMod<T>
SubMod<T>
NegMod<T>
MulMod<T>
MulInvOp<T>
MadMod<T>

// element wise coeff mod multiplication
// general mulmod
void MulModPoly<T>(
    int n_polys,  // n of polys
    int n_primes,        // n of primes
    int poly_order,      // poly order
    const T* operand1,     // operand1, size = (n_polys*n_primes*poly_order)
    const T* operand2,     // operand2, size = (n_polys*n_primes*poly_order)
    T* result,          // output
    const T* modulus    // primes
    );

// precomputed mod inverse
void MulModPoly<T>(
    int n_polys,  // n of polys
    int n_primes,        // n of primes
    int poly_order,      // poly order
    const T* operand1,     // operand1, size = (n_polys*n_primes*poly_order)
    const T* operand2,     // operand2, size = (n_polys*n_primes*poly_order)
    T* result,          // output
    const T* modulus,    // primes
    const T* modulus_inverse // 2^(2T)/modulus
    );

// precomputed operan2 inverse
void MulModPoly<T>(
    int n_polys,  // n of polys
    int n_primes,        // n of primes
    int poly_order,      // poly order
    const T* operand1,     // operand1, size = (n_polys*n_primes*poly_order)
    const T* operand2,     // operand2, size = (n_polys*n_primes*poly_order)
    const T* op2_quo,     // 2^T*operand2/mod
    T* result,          // output
    const T* modulus    // primes
    );



// NTT/iNTT
// iNTT has similar interface
int NTT<T>(int n_polys,  // n of polys
    int n_primes,        // n of primes
    int poly_order,      // poly order
    const T* input,     // input, size = (n_polys*n_primes*poly_order)
    T* output,          // output, size = (n_polys*n_primes*poly_order) for weighted transform and *2 by na�ve NTT, 
    bool reverse_order,   // input order, input inverse->output normal order; input normal->output reverse order
    bool weighted,        // weighted or na�ve NTT transforms, na�ve requires 2X output
    const T* modulus,    // primes
    const T* roots_op,   // array of roots of unity
    const T* roots_quo,  // array of roots/prime for faster mulmod
    const T* scalar_op = nullptr, // scaler 
    const T* scalar_quo = nullptr) // scaler/prime;

// coefficient wise multiplication of multiple pairs of polynomials

// input polynomials and result each are represented in memory as compact 3D tensor
// n_polys x rns_base_size x poly_order 

// flags define modulo operations’ supporting info
// flags: 0 – only modulo is present
// flags: 1 – modulo inverse is present (modulo_inverse != nullptr)                            = (2^ (bitsize(T)*2)) / modulo
// used as an argument to the barret reducton for generic mul_mod op.
// flags: 2 – scaled operand 2 divided by modulo is present (op2_quo != nullptr)    = (2^ bitsize(T) * op2) / modulo
// speedup mul_mod op

// primes bit size condition: 
// true : bitsize <= bitsizeof(T) -4 
// false: bitsize <= bitsizeof(T) -1 

// returns an event associated with the launch
// the even is going to be raised after the interface completes. 
Event EwsMulModPoly<T>(
    uint64_t flags,
    bool prime_bit_size,
    int n_polys,  // n of polys
    int rns_base_size,        // n of primes
    int poly_order,      // poly order
    const T * operand1,     // operand1
    const T * operand2,     // operand2
    T * result,          // output
    const T * modulo,    // primes
    const T* modulo_inverse = nullptr, // 2^(2T)/modulus
    const T* op2_quo = nullptr,   // 2^T*operand2/mod
    uint64_t device_id,                                 // device the interface is running on
    uint64_t stream_id,                                // command stream or queue 
    const Event * events             // list of events this interface has to wait on before the launch
    );

//There 2 other MulPoly interfaces are needed:
//1.	Standard polynomial multiplication.
//              It can be implemented as a sequences:
//              NTT, EwsMulModPoly, iNTT 
//2.	HE Mul.
//It might be naivly implemented as 
//(a,b) x (c,d) = (a*c, a*d+b*c, b*d),
//Where a,b,c,d are all polynomials as defined at the top and * = EwsMulModPoly, + = element-wise mod addition.


// primes bit size condition: 
// bitsize <= bitsizeof(T) -4 
// bitsize <= bitsizeof(T) -1 

// transform an array of values from positional to rns representation 
Event CRT<T>(
uint64_t flags, // 0:  bitsize <= bitsizeof(T) -4; 1: bitsize <= bitsizeof(T) -1
const std::vector<T> & values,
 T range,
 const std::vector<T> & rns_base, 
std::vector<std::vector<T>> &rns_representations,
uint64_t device_id,                                 // device the interface is running on
 uint64_t stream_id,                                // command stream or queue 
 const Event * events             // list of events this interface has to wait on before the launch
); 
// inverse transform
Event iCRT<T>(
uint64_t flags, // 0:  bitsize <= bitsizeof(T) -4; 1: bitsize <= bitsizeof(T) -1
std::vector<T> & values,
 T range,
 const std::vector<T> & rns_base, 
const std::vector<std::vector<T>> &rns_representations,
 uint64_t device_id,                                 // device the interface is running on
 uint64_t stream_id,                                // command stream or queue 
 const Event * events             // list of events this interface has to wait on before the launch
); 


