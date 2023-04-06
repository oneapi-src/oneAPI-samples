/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#include "Plaintext.h"

Plaintext::Plaintext() : mx(nullptr), N(0), slots(0), l(0) {}

Plaintext::Plaintext(uint64_t* mx, long N, long slots, long l) : mx(mx), N(N), slots(slots), l(l) {}

Plaintext::Plaintext(const Plaintext& ptxt) : N(ptxt.N), slots(ptxt.slots), l(ptxt.l){
	mx = new uint64_t[N * l];
	for (long i = 0; i < N * l; ++i) {
		mx[i] = ptxt.mx[i];
	}
}

Plaintext& Plaintext::operator=(const Plaintext& o) {
	if(this == &o) return *this; // handling of self assignment, thanks for your advice, arul.
	delete[] mx;
	N = o.N;
	l = o.l;
	slots = o.slots;
	mx = new uint64_t[N * l];
	for (long i = 0; i < N * l; ++i) {
		mx[i] = o.mx[i];
	}
	return *this;
}
