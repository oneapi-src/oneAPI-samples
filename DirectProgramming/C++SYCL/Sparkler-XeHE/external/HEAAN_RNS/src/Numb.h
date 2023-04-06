/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#ifndef HEAANNTT_NUMB_H_
#define HEAANNTT_NUMB_H_

#include <iostream>
#include <vector>
#include <set>
#include <math.h>
#include <stdint.h>

#include "Common.h"

using namespace std;

void negate(uint64_t& r, uint64_t a);

void addMod(uint64_t& r, uint64_t a, uint64_t b, uint64_t p);

void addModAndEqual(uint64_t& a, uint64_t b, uint64_t p);

void subMod(uint64_t& r, uint64_t a, uint64_t b, uint64_t p);

void subModAndEqual(uint64_t& a, uint64_t b, uint64_t p);

void mulMod(uint64_t& r, uint64_t a, uint64_t b, uint64_t p);

void mulModBarrett(uint64_t& r, uint64_t a, uint64_t b, uint64_t p, uint64_t pr, long twok);

void modBarrett(uint64_t &r, uint64_t a, uint64_t m, uint64_t mr, long twok);

void modBarrett(uint64_t &r, unsigned __int128 a, uint64_t m, uint64_t mr, long twok);

void mulModAndEqual(uint64_t& a, uint64_t b, uint64_t p);

uint64_t invMod(uint64_t x, uint64_t p);

uint64_t powMod(uint64_t x, uint64_t y, uint64_t p);

uint64_t inv(uint64_t x);

uint64_t pow(uint64_t x, uint64_t y);

uint32_t bitReverse(uint32_t x);

uint64_t gcd(uint64_t a, uint64_t b);

long gcd(long a, long b);

void findPrimeFactors(set<uint64_t> &s, uint64_t number);

uint64_t findPrimitiveRoot(uint64_t m);

uint64_t findMthRootOfUnity(uint64_t M, uint64_t p);

bool primeTest(uint64_t p);

#endif
