/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#include "Numb.h"

void negate(uint64_t &r, uint64_t a) {
	r = -a;
}

void addMod(uint64_t &r, uint64_t a, uint64_t b, uint64_t m) {
	r = (a + b) % m;
}

void subMod(uint64_t &r, uint64_t a, uint64_t b, uint64_t m) {
	r = b % m;
	r = (a + m - r) % m;
}

void mulMod(uint64_t &r, uint64_t a, uint64_t b, uint64_t m) {
	unsigned __int128 mul = static_cast<unsigned __int128>(a % m) * (b % m);
	mul %= static_cast<unsigned __int128>(m);
	r = static_cast<uint64_t>(mul);
}

void mulModBarrett(uint64_t& r, uint64_t a, uint64_t b, uint64_t p, uint64_t pr, long twok) {
	unsigned __int128 mul = static_cast<unsigned __int128>(a % p) * (b % p);
	modBarrett(r, mul, p, pr, twok);
}

void modBarrett(uint64_t &r, uint64_t a, uint64_t m, uint64_t mr, long twok) {
	unsigned __int128 tmp = static_cast<unsigned __int128>(a) * mr;
	tmp >>= twok;
	tmp *= m;
	tmp = a - tmp;
	r = static_cast<uint64_t>(tmp);
	if(r < m) {
		return;
	}
	else {
		r -= m;
		return;
	}

}

void modBarrett(uint64_t &r, unsigned __int128 a, uint64_t m, uint64_t mr, long twok) {
	uint64_t atop, abot;
	abot = static_cast<uint64_t>(a);
	atop = static_cast<uint64_t>(a >> 64);
	unsigned __int128 tmp = static_cast<unsigned __int128>(abot) * mr;
	tmp >>= 64;
	tmp += static_cast<unsigned __int128>(atop) * mr;
	tmp >>= twok - 64;
	tmp *= m;
	tmp = a - tmp;
	r = static_cast<uint64_t>(tmp);
	if(r >= m) r-= m;
}

uint64_t invMod(uint64_t x, uint64_t m) {
	uint64_t temp = x % m;
	if (gcd(temp, m) != 1) {
		throw invalid_argument("Inverse doesn't exist!!!");
	} else {
		return powMod(temp, m - 2, m);
	}
}

uint64_t powMod(uint64_t x, uint64_t y, uint64_t modulus) {
	uint64_t res = 1;
	while (y > 0) {
		if (y & 1) {
			mulMod(res, res, x, modulus);
		}
		y = y >> 1;
		mulMod(x, x, x, modulus);
	}
	return res;
}

uint64_t inv(uint64_t x) {
	return pow(x, static_cast<uint64_t>(-1));
}

uint64_t pow(uint64_t x, uint64_t y) {
	uint64_t res = 1;
	while (y > 0) {
		if (y & 1) {
			res *= x;
		}
		y = y >> 1;
		x *= x;
	}
	return res;
}

uint32_t bitReverse(uint32_t x) {
	x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
	x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
	x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
	x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
	return ((x >> 16) | (x << 16));
}

uint64_t gcd(uint64_t a, uint64_t b) {
	if (a == 0) {
		return b;
	}
	return gcd(b % a, a);
}

long gcd(long a, long b) {
	if (a == 0) {
		return b;
	}
	return gcd(b % a, a);
}

void findPrimeFactors(set<uint64_t> &s, uint64_t number) {
	while (number % 2 == 0) {
		s.insert(2);
		number /= 2;
	}
	for (uint64_t i = 3; i < sqrt(number); i++) {
		while (number % i == 0) {
			s.insert(i);
			number /= i;
		}
	}
	if (number > 2) {
		s.insert(number);
	}
}

uint64_t findPrimitiveRoot(uint64_t modulus) {
	set<uint64_t> s;
	uint64_t phi = modulus - 1;
	findPrimeFactors(s, phi);
	for (uint64_t r = 2; r <= phi; r++) {
		bool flag = false;
		for (auto it = s.begin(); it != s.end(); it++) {
			if (powMod(r, phi / (*it), modulus) == 1) {
				flag = true;
				break;
			}
		}
		if (flag == false) {
			return r;
		}
	}
	return -1;
}

uint64_t findMthRootOfUnity(uint64_t M, uint64_t mod) {
    uint64_t res;
    res = findPrimitiveRoot(mod);
    if((mod - 1) % M == 0) {
        uint64_t factor = (mod - 1) / M;
        res = powMod(res, factor, mod);
        return res;
    }
    else {
        return -1;
    }
}

// Miller-Rabin Prime Test //
bool primeTest(uint64_t p) {
	if(p < 2) return false;
	if(p != 2 && p % 2 == 0) return false;
	uint64_t s = p - 1;
	while(s % 2 == 0) {
		s /= 2;
	}
	for(long i = 0; i < 200; i++) {
		uint64_t temp1 = rand();
		temp1  = (temp1 << 32) | rand();
		temp1 = temp1 % (p - 1) + 1;
		uint64_t temp2 = s;
		uint64_t mod = powMod(temp1,temp2,p);
		while (temp2 != p - 1 && mod != 1 && mod != p - 1) {
			mulMod(mod, mod, mod, p);
		    temp2 *= 2;
		}
		if (mod != p - 1 && temp2 % 2 == 0) return false;
	}
	return true;
}
