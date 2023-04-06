/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#include "Scheme.h"

Scheme::Scheme(Context& context) : context(context) {}

Scheme::Scheme(SecretKey& secretKey, Context& context) : context(context) {
	addEncKey(secretKey);
	addMultKey(secretKey);
}

void Scheme::addEncKey(SecretKey& secretKey) {
	uint64_t* ex = new uint64_t[context.L << context.logN]();
	uint64_t* ax = new uint64_t[context.L << context.logN]();
	uint64_t* bx = new uint64_t[context.L << context.logN]();

	context.sampleUniform(ax, context.L);

	context.sampleGauss(ex, context.L);
	context.NTTAndEqual(ex, context.L);

	context.mul(bx, ax, secretKey.sx, context.L);
	context.sub2AndEqual(ex, bx, context.L);

	delete[] ex;

	keyMap.insert(pair<long, Key>(ENCRYPTION, Key(ax, bx)));
}

void Scheme::addMultKey(SecretKey& secretKey) {
	uint64_t* ex = new uint64_t[(context.L + context.K) << context.logN]();
	uint64_t* ax = new uint64_t[(context.L + context.K) << context.logN]();
	uint64_t* bx = new uint64_t[(context.L + context.K) << context.logN]();
	uint64_t* sxsx = new uint64_t[(context.L + context.K) << context.logN]();

	context.mul(sxsx, secretKey.sx, secretKey.sx, context.L);

	context.evalAndEqual(sxsx, context.L);

	context.sampleGauss(ex, context.L, context.K);
	context.NTTAndEqual(ex, context.L, context.K);

	context.addAndEqual(ex, sxsx, context.L);

	context.sampleUniform(ax, context.L, context.K);
	context.mul(bx, ax, secretKey.sx, context.L, context.K);
	context.sub2AndEqual(ex, bx, context.L, context.K);

	delete[] ex;
	delete[] sxsx;

	keyMap.insert(pair<long, Key>(MULTIPLICATION, Key(ax, bx)));
}

void Scheme::addConjKey(SecretKey& secretKey) {
	uint64_t* ex = new uint64_t[(context.L + context.K) << context.logN]();
	uint64_t* ax = new uint64_t[(context.L + context.K) << context.logN]();
	uint64_t* bx = new uint64_t[(context.L + context.K) << context.logN]();
	uint64_t* sxconj = new uint64_t[(context.L + context.K) << context.logN]();

	context.conjugate(sxconj, secretKey.sx, context.L);
	context.evalAndEqual(sxconj, context.L);

	context.sampleGauss(ex, context.L, context.K);
	context.NTTAndEqual(ex, context.L, context.K);

	context.addAndEqual(ex, sxconj, context.L);

	context.sampleUniform(ax, context.L, context.K);
	context.mul(bx, ax, secretKey.sx, context.L, context.K);
	context.sub2AndEqual(ex, bx, context.L, context.K);

	delete[] ex;
	delete[] sxconj;

	keyMap.insert(pair<long, Key>(CONJUGATION, Key(ax, bx)));
}

void Scheme::addLeftRotKey(SecretKey& secretKey, long rot) {
	uint64_t* ex = new uint64_t[(context.L + context.K) << context.logN]();
	uint64_t* ax = new uint64_t[(context.L + context.K) << context.logN]();
	uint64_t* bx = new uint64_t[(context.L + context.K) << context.logN]();
	uint64_t* sxrot = new uint64_t[(context.L + context.K) << context.logN]();

	context.leftRot(sxrot, secretKey.sx, context.L, rot);
	context.evalAndEqual(sxrot, context.L);

	context.sampleGauss(ex, context.L, context.K);
	context.NTTAndEqual(ex, context.L, context.K);

	context.addAndEqual(ex, sxrot, context.L);

	context.sampleUniform(ax, context.L, context.K);
	context.mul(bx, ax, secretKey.sx, context.L, context.K);
	context.sub2AndEqual(ex, bx, context.L, context.K);

	delete[] ex;
	delete[] sxrot;

	leftRotKeyMap.insert(pair<long, Key>(rot, Key(ax, bx)));
}

void Scheme::addLeftRotKeys(SecretKey& secretKey) {
	for (long i = 0; i < context.logNh; ++i) {
		long idx = 1 << i;
		if(leftRotKeyMap.find(idx) == leftRotKeyMap.end()) {
			addLeftRotKey(secretKey, idx);
		}
	}
}

void Scheme::addRightRotKeys(SecretKey& secretKey) {
	for (long i = 0; i < context.logNh; ++i) {
		long idx = context.Nh - (1 << i);
		if(leftRotKeyMap.find(idx) == leftRotKeyMap.end()) {
			addLeftRotKey(secretKey, idx);
		}
	}
}

Plaintext Scheme::encode(double* v, long slots, long l) {
	uint64_t* m = new uint64_t[l << context.logN]();
	context.encode(m, v, slots, l);
	return Plaintext(m, context.N, slots, l);
}

Plaintext Scheme::encode(complex<double>* v, long slots, long l) {
	uint64_t* m = new uint64_t[l << context.logN]();
	context.encode(m, v, slots, l);
	return Plaintext(m, context.N, slots, l);
}

Plaintext Scheme::encodeSingle(complex<double> val, long l) {
	uint64_t* m = new uint64_t[l << context.logN]();
	context.encodeSingle(m, val, l);
	return Plaintext(m, context.N, 1, l);
}

complex<double>* Scheme::decode(Plaintext& msg) {
	complex<double>* res = new complex<double>[msg.slots]();
	context.decode(msg.mx, res, msg.slots, msg.l);
	return res;
}

complex<double> Scheme::decodeSingle(Plaintext& msg) {
	complex<double> res;
	context.decodeSingle(msg.mx, res, msg.l);
	return res;
}

Ciphertext Scheme::encryptMsg(SecretKey& secretkey, Plaintext& message) {
	Ciphertext res;
	return res;
}


Ciphertext Scheme::encryptMsg(Plaintext& message) {
	Key key = keyMap.at(ENCRYPTION);

	uint64_t* ax = new uint64_t[message.l << context.logN]();
	uint64_t* bx = new uint64_t[message.l << context.logN]();
	uint64_t* vx = new uint64_t[message.l << context.logN]();
	uint64_t* ex = new uint64_t[message.l << context.logN]();

	context.sampleZO(vx, context.Nh, message.l);
	context.NTTAndEqual(vx, message.l);

	context.mul(ax, vx, key.ax, message.l);

	context.sampleGauss(ex, message.l);
	context.NTTAndEqual(ex, message.l);

	context.addAndEqual(ax, ex, message.l);

	context.mul(bx, vx, key.bx, message.l);

	context.sampleGauss(ex, message.l);
	context.NTTAndEqual(ex, message.l);

	context.addAndEqual(bx, ex, message.l);
	context.addAndEqual(bx, message.mx, message.l);

	return Ciphertext(ax, bx, context.N, message.slots, message.l);
}

Plaintext Scheme::decryptMsg(SecretKey& secretKey, Ciphertext& cipher) {
	uint64_t* mx = new uint64_t[context.N]();
	context.mul(mx, cipher.ax, secretKey.sx, 1);
	context.addAndEqual(mx, cipher.bx, 1);

	return Plaintext(mx, context.N, cipher.slots, 1);
}

Ciphertext Scheme::encrypt(double* vals, long slots, long l) {
	Plaintext msg = encode(vals, slots, l);
	return encryptMsg(msg);
}

Ciphertext Scheme::encrypt(complex<double>* vals, long slots, long l) {
	Plaintext msg = encode(vals, slots, l);
	return encryptMsg(msg);
}

Ciphertext Scheme::encryptSingle(complex<double> val, long l) {
	Plaintext msg = encodeSingle(val, l);
	return encryptMsg(msg);
}

complex<double>* Scheme::decrypt(SecretKey& secretKey, Ciphertext& cipher) {
	Plaintext msg = decryptMsg(secretKey, cipher);
	return decode(msg);
}

complex<double> Scheme::decryptSingle(SecretKey& secretKey, Ciphertext& cipher) {
	Plaintext msg = decryptMsg(secretKey, cipher);
	return decodeSingle(msg);
}

Ciphertext Scheme::negate(Ciphertext& cipher) {
	uint64_t* axres = new uint64_t[cipher.l << context.logN];
	uint64_t* bxres = new uint64_t[cipher.l << context.logN];

	context.negate(axres, cipher.ax, cipher.l);
	context.negate(bxres, cipher.bx, cipher.l);

	return Ciphertext(axres, bxres, context.N, cipher.slots, cipher.l);
}

void Scheme::negateAndEqual(Ciphertext& cipher) {
	long shift = 0;
	for (long i = 0; i < cipher.l; ++i) {
		context.qiNegateAndEqual(cipher.ax + shift, i);
		context.qiNegateAndEqual(cipher.bx + shift, i);
		shift += context.N;
	}
}

Ciphertext Scheme::add(Ciphertext& cipher1, Ciphertext& cipher2) {
	uint64_t* axres = new uint64_t[cipher1.l << context.logN];
	uint64_t* bxres = new uint64_t[cipher1.l << context.logN];

	context.add(axres, cipher1.ax, cipher2.ax, cipher1.l);
	context.add(bxres, cipher1.bx, cipher2.bx, cipher1.l);

	return Ciphertext(axres, bxres, context.N, cipher1.slots, cipher1.l);
}

void Scheme::addAndEqual(Ciphertext& cipher1, Ciphertext& cipher2) {
	if(cipher1.l != cipher2.l) {
		throw invalid_argument("Ciphertexts are not comparable");
	}

	context.addAndEqual(cipher1.ax, cipher2.ax, cipher1.l);
	context.addAndEqual(cipher1.bx, cipher2.bx, cipher1.l);
}

Ciphertext Scheme::sub(Ciphertext& cipher1, Ciphertext& cipher2) {
	if(cipher1.l != cipher2.l) {
		throw invalid_argument("Ciphertexts are not comparable");
	}

	uint64_t* axres = new uint64_t[cipher1.l << context.logN];
	uint64_t* bxres = new uint64_t[cipher1.l << context.logN];

	context.sub(axres, cipher1.ax, cipher2.ax, cipher1.l);
	context.sub(bxres, cipher1.bx, cipher2.bx, cipher1.l);

	return Ciphertext(axres, bxres, context.N, cipher1.slots, cipher1.l);
}

void Scheme::subAndEqual(Ciphertext& cipher1, Ciphertext& cipher2) {
	if(cipher1.l != cipher2.l) {
		throw invalid_argument("Ciphertexts are not comparable");
	}

	context.subAndEqual(cipher1.ax, cipher2.ax, cipher1.l);
	context.subAndEqual(cipher1.bx, cipher2.bx, cipher1.l);
}

void Scheme::sub2AndEqual(Ciphertext& cipher1, Ciphertext& cipher2) {
	if(cipher1.l != cipher2.l) {
		throw invalid_argument("Ciphertexts are not comparable");
	}

	context.sub2AndEqual(cipher1.ax, cipher2.ax, cipher1.l);
	context.sub2AndEqual(cipher1.bx, cipher2.bx, cipher1.l);
}

Ciphertext Scheme::mult(Ciphertext& cipher1, Ciphertext& cipher2) {
	uint64_t* axbx1 = new uint64_t[cipher1.l << context.logN]();
	uint64_t* axbx2 = new uint64_t[cipher1.l << context.logN]();

	uint64_t* axax = new uint64_t[cipher1.l << context.logN]();
	uint64_t* bxbx = new uint64_t[cipher1.l << context.logN]();

	uint64_t* axmult = new uint64_t[(cipher1.l + context.K) << context.logN]();
	uint64_t* bxmult = new uint64_t[(cipher1.l + context.K) << context.logN]();

	context.add(axbx1, cipher1.ax, cipher1.bx, cipher1.l);
	context.add(axbx2, cipher2.ax, cipher2.bx, cipher2.l);
	context.mulAndEqual(axbx1, axbx2, cipher1.l);
	context.mul(bxbx, cipher1.bx, cipher2.bx, cipher1.l);
	context.mul(axax, cipher1.ax, cipher2.ax, cipher1.l);

	context.raiseAndEqual(axax, cipher1.l);

	Key key = keyMap.at(MULTIPLICATION);

	context.mulKey(axmult, axax, key.ax, cipher1.l);
	context.mulKey(bxmult, axax, key.bx, cipher1.l);

	context.backAndEqual(axmult, cipher1.l);
	context.backAndEqual(bxmult, cipher1.l);

	context.addAndEqual(axmult, axbx1, cipher1.l);
	context.subAndEqual(axmult, bxbx, cipher1.l);
	context.subAndEqual(axmult, axax, cipher1.l);
	context.addAndEqual(bxmult, bxbx, cipher1.l);

	delete[] axax;
	delete[] bxbx;
	delete[] axbx1;
	delete[] axbx2;

	return Ciphertext(axmult, bxmult, context.N, cipher1.slots, cipher1.l);
}

void Scheme::multAndEqual(Ciphertext& cipher1, Ciphertext& cipher2) {

	uint64_t* axbx1 = new uint64_t[cipher1.l << context.logN]();
	uint64_t* axbx2 = new uint64_t[cipher1.l << context.logN]();

	uint64_t* axax = new uint64_t[cipher1.l << context.logN]();
	uint64_t* bxbx = new uint64_t[cipher1.l << context.logN]();

	context.add(axbx1, cipher1.ax, cipher1.bx, cipher1.l); // ax1 + bx1 mod P, 0 mod Q
	context.add(axbx2, cipher2.ax, cipher2.bx, cipher2.l); // ax2 + bx2 mod P, 0 mod Q

	context.mulAndEqual(axbx1, axbx2, cipher1.l); // (ax1 + bx1) * (ax2 + bx2) mod P, 0 mod Q

	context.mul(bxbx, cipher1.bx, cipher2.bx, cipher1.l); // bx1 * bx2 mod P, 0 mod Q
	context.mul(axax, cipher1.ax, cipher2.ax, cipher1.l); // ax1 * ax2 mod P, 0 mod Q
	context.raiseAndEqual(axax, cipher1.l); // ax1 * ax2 mod P, ax1 * ax2 + e * P mod Q

	Key key = keyMap.at(MULTIPLICATION); // kbx - kax * sx = (sxsx * Q + ex mod P, ex mod Q)

	delete[] cipher1.ax;
	delete[] cipher1.bx;

	cipher1.ax = new uint64_t[(cipher1.l + context.K) << context.logN]();
	cipher1.bx = new uint64_t[(cipher1.l + context.K) << context.logN]();

	context.mulKey(cipher1.ax, axax, key.ax, cipher1.l);
	context.mulKey(cipher1.bx, axax, key.bx, cipher1.l);

	context.backAndEqual(cipher1.ax, cipher1.l);
	context.backAndEqual(cipher1.bx, cipher1.l);

	context.addAndEqual(cipher1.ax, axbx1, cipher1.l);
	context.subAndEqual(cipher1.ax, bxbx, cipher1.l);
	context.subAndEqual(cipher1.ax, axax, cipher1.l);
	context.addAndEqual(cipher1.bx, bxbx, cipher1.l);

	delete[] axax;
	delete[] bxbx;
	delete[] axbx1;
	delete[] axbx2;
}


Ciphertext Scheme::square(Ciphertext& cipher) {
	uint64_t* axbx = new uint64_t[cipher.l << context.logN]();

	uint64_t* axax = new uint64_t[cipher.l << context.logN]();
	uint64_t* bxbx = new uint64_t[cipher.l << context.logN]();

	uint64_t* axmult = new uint64_t[(cipher.l + context.K) << context.logN]();
	uint64_t* bxmult = new uint64_t[(cipher.l + context.K) << context.logN]();

	context.add(axbx, cipher.ax, cipher.bx, cipher.l); // ax1 + bx1 mod P, 0 mod Q

	context.squareAndEqual(axbx, cipher.l); // (ax1 + bx1) * (ax2 + bx2) mod P, 0 mod Q
	context.square(bxbx, cipher.bx, cipher.l);
	context.square(axax, cipher.ax, cipher.l);

	context.raiseAndEqual(axax, cipher.l);

	Key key = keyMap.at(MULTIPLICATION);

	context.mulKey(axmult, axax, key.ax, cipher.l);
	context.mulKey(bxmult, axax, key.bx, cipher.l);

	context.backAndEqual(axmult, cipher.l);
	context.backAndEqual(bxmult, cipher.l);

	context.addAndEqual(axmult, axbx, cipher.l);
	context.subAndEqual(axmult, bxbx, cipher.l);
	context.subAndEqual(axmult, axax, cipher.l);
	context.addAndEqual(bxmult, bxbx, cipher.l);

	delete[] axax;
	delete[] bxbx;
	delete[] axbx;

	return Ciphertext(axmult, bxmult, context.N, cipher.slots, cipher.l);
}

void Scheme::squareAndEqual(Ciphertext& cipher) {
	uint64_t* axbx = new uint64_t[cipher.l << context.logN]();
	uint64_t* axax = new uint64_t[cipher.l << context.logN]();
	uint64_t* bxbx = new uint64_t[cipher.l << context.logN]();

	context.add(axbx, cipher.ax, cipher.bx, cipher.l); // ax1 + bx1 mod P, 0 mod Q

	context.squareAndEqual(axbx, cipher.l); // (ax1 + bx1) * (ax2 + bx2) mod P, 0 mod Q
	context.square(bxbx, cipher.bx, cipher.l);
	context.square(axax, cipher.ax, cipher.l);

	context.raiseAndEqual(axax, cipher.l);

	delete[] cipher.ax;
	delete[] cipher.bx;

	cipher.ax = new uint64_t[(cipher.l + context.K) << context.logN]();
	cipher.bx = new uint64_t[(cipher.l + context.K) << context.logN]();

	Key key = keyMap.at(MULTIPLICATION);

	context.mulKey(cipher.ax, axax, key.ax, cipher.l);
	context.mulKey(cipher.bx, axax, key.bx, cipher.l);

	context.backAndEqual(cipher.ax, cipher.l);
	context.backAndEqual(cipher.bx, cipher.l);

	context.addAndEqual(cipher.ax, axbx, cipher.l);
	context.subAndEqual(cipher.ax, bxbx, cipher.l);
	context.subAndEqual(cipher.ax, axax, cipher.l);
	context.addAndEqual(cipher.bx, bxbx, cipher.l);

	delete[] axax;
	delete[] bxbx;
	delete[] axbx;
}

Ciphertext Scheme::imult(Ciphertext& cipher) {
	uint64_t* ax = new uint64_t[cipher.l << context.logN]();
	uint64_t* bx = new uint64_t[cipher.l << context.logN]();

	context.mulByMonomial(ax, cipher.ax, cipher.l, context.Nh);
	context.mulByMonomial(bx, cipher.bx, cipher.l, context.Nh);

	return Ciphertext(ax, bx, context.N, cipher.slots, cipher.l);
}

void Scheme::imultAndEqual(Ciphertext& cipher) {
	//TODO implement method
}

Ciphertext Scheme::idiv(Ciphertext& cipher) {
	//TODO implement method
	Ciphertext res;
	return res;

}

void Scheme::idivAndEqual(Ciphertext& cipher) {
	//TODO implement method
}

Ciphertext Scheme::addConst(Ciphertext& cipher, double cnst) {
	uint64_t tmpr = abs(cnst) * context.p;
	uint64_t* ax = new uint64_t[cipher.l << context.logN]();
	uint64_t* bx = new uint64_t[cipher.l << context.logN]();
	copy(cipher.ax, cipher.ax + (cipher.l << context.logN), ax);

	if(cnst >= 0) {
		context.addConst(bx, cipher.bx, tmpr, cipher.l);
	} else {
		context.subConst(bx, cipher.bx, tmpr, cipher.l);
	}

	return Ciphertext(ax, bx, context.N, cipher.slots, cipher.l);

}

Ciphertext Scheme::addConst(Ciphertext& cipher, complex<double> cnst) {
	//TODO implement method
	Ciphertext res;
	return res;

}

void Scheme::addConstAndEqual(Ciphertext& cipher, double cnst) {
	uint64_t tmpr = abs(cnst) * context.p;
	if(cnst >= 0) {
		context.addConstAndEqual(cipher.bx, tmpr, cipher.l);
	} else {
		context.subConstAndEqual(cipher.bx, tmpr, cipher.l);
	}
}

void Scheme::addConstAndEqual(Ciphertext& cipher, complex<double> cnst) {
	//TODO implement method
}

void Scheme::addPcAndEqual(Ciphertext& cipher) {
	context.addAndEqual(cipher.bx, context.pccoeff, cipher.l);
}

void Scheme::addP2AndEqual(Ciphertext& cipher) {
	context.addAndEqual(cipher.bx, context.p2coeff, cipher.l);
}

void Scheme::addP2hAndEqual(Ciphertext& cipher) {
	context.addAndEqual(cipher.bx, context.p2hcoeff, cipher.l);
}

Ciphertext Scheme::multByConst(Ciphertext& cipher, double cnst) {
	uint64_t tmpr = abs(cnst) * context.p;

	uint64_t* ax = new uint64_t[cipher.l << context.logN]();
	uint64_t* bx = new uint64_t[cipher.l << context.logN]();

	context.mulConst(ax, cipher.ax, tmpr, cipher.l);
	context.mulConst(bx, cipher.bx, tmpr, cipher.l);

	if(cnst < 0) {
		context.negateAndEqual(ax, cipher.l);
		context.negateAndEqual(bx, cipher.l);
	}
	return Ciphertext(ax, bx, context.N, cipher.slots, cipher.l);
}

Ciphertext Scheme::multByConst(Ciphertext& cipher, complex<double> cnst) {
	uint64_t tmpr = cnst.real() * context.p;
	uint64_t tmpi = cnst.imag() * context.p;

	uint64_t* ax = new uint64_t[cipher.l << context.logN]();
	uint64_t* bx = new uint64_t[cipher.l << context.logN]();
	uint64_t* axi = new uint64_t[cipher.l << context.logN]();
	uint64_t* bxi = new uint64_t[cipher.l << context.logN]();

	context.mulByMonomial(axi, cipher.ax, cipher.l, context.Nh);
	context.mulByMonomial(bxi, cipher.bx, cipher.l, context.Nh);

	context.mulConst(ax, cipher.ax, tmpr, cipher.l);
	context.mulConst(bx, cipher.bx, tmpr, cipher.l);

	context.mulConstAndEqual(axi, tmpi, cipher.l);
	context.mulConstAndEqual(bxi, tmpi, cipher.l);

	context.addAndEqual(ax, axi, cipher.l);
	context.addAndEqual(bx, bxi, cipher.l);

	return Ciphertext(ax, bx, context.N, cipher.slots, cipher.l);
}

Ciphertext Scheme::multByConstVec(Ciphertext& cipher, double* cnstVec, long slots) {
	uint64_t* ax = new uint64_t[cipher.l << context.logN]();
	uint64_t* bx = new uint64_t[cipher.l << context.logN]();
	uint64_t* mx = new uint64_t[cipher.l << context.logN]();

	context.encode(mx, cnstVec, slots, cipher.l);
	context.mul(ax, cipher.ax, mx, cipher.l);
	context.mul(bx, cipher.bx, mx, cipher.l);
	delete[] mx;
	return Ciphertext(ax, bx, context.N, cipher.slots, cipher.l);
}

Ciphertext Scheme::multByConstVec(Ciphertext& cipher, complex<double>* cnstVec, long slots) {
	uint64_t* ax = new uint64_t[cipher.l << context.logN]();
	uint64_t* bx = new uint64_t[cipher.l << context.logN]();
	uint64_t* mx = new uint64_t[cipher.l << context.logN]();

	context.encode(mx, cnstVec, slots, cipher.l);
	context.mul(ax, cipher.ax, mx, cipher.l);
	context.mul(bx, cipher.bx, mx, cipher.l);
	delete[] mx;
	return Ciphertext(ax, bx, context.N, cipher.slots, cipher.l);
}

void Scheme::multByConstAndEqual(Ciphertext& cipher, double cnst) {
	uint64_t tmpr = abs(cnst) * context.p;

	uint64_t* ax = new uint64_t[cipher.l << context.logN]();
	uint64_t* bx = new uint64_t[cipher.l << context.logN]();

	context.mulConstAndEqual(cipher.ax, tmpr, cipher.l);
	context.mulConstAndEqual(cipher.bx, tmpr, cipher.l);

	if(cnst < 0) {
		context.negateAndEqual(cipher.ax, cipher.l);
		context.negateAndEqual(cipher.bx, cipher.l);
	}
}

void Scheme::multByConstAndEqual(Ciphertext& cipher, complex<double> cnst) {
}

void Scheme::multByPolyAndEqual(Ciphertext& cipher, uint64_t* poly) {
	context.mulAndEqual(cipher.ax, poly, cipher.l);
	context.mulAndEqual(cipher.bx, poly, cipher.l);
}

Ciphertext Scheme::multByMonomial(Ciphertext& cipher, long mdeg) {
	//TODO implement method
	Ciphertext res;
	return res;
}

void Scheme::multByMonomialAndEqual(Ciphertext& cipher, long mdeg) {
	//TODO implement method
}


Ciphertext Scheme::reScaleBy(Ciphertext& cipher, long dl) {
	//TODO implement method
	Ciphertext res;
	return res;
}

void Scheme::reScaleByAndEqual(Ciphertext& cipher, long dl) {
	for (long i = 0; i < dl; ++i) {
		context.reScaleAndEqual(cipher.ax, cipher.l);
		context.reScaleAndEqual(cipher.bx, cipher.l);
		cipher.l -= 1;
	}
}

Ciphertext Scheme::reScaleTo(Ciphertext& cipher, long l) {
	long dl = cipher.l - l;
	return reScaleBy(cipher, dl);
}

void Scheme::reScaleToAndEqual(Ciphertext& cipher, long l) {
	long dl = cipher.l - l;
	reScaleByAndEqual(cipher, dl);
}

Ciphertext Scheme::modDownBy(Ciphertext& cipher, long dl) {
	uint64_t* ax = context.modDown(cipher.ax, cipher.l, dl);
	uint64_t* bx = context.modDown(cipher.bx, cipher.l, dl);
	return Ciphertext(ax, bx, context.N, cipher.slots, cipher.l - dl);
}

void Scheme::modDownByAndEqual(Ciphertext& cipher, long dl) {
	context.modDownAndEqual(cipher.ax, cipher.l, dl);
	context.modDownAndEqual(cipher.bx, cipher.l, dl);
	cipher.l -= dl;
}

Ciphertext Scheme::modDownTo(Ciphertext& cipher, long l) {
	long dl = cipher.l - l;
	return modDownBy(cipher, dl);
}

void Scheme::modDownToAndEqual(Ciphertext& cipher, long l) {
	long dl = cipher.l - l;
	modDownByAndEqual(cipher, dl);
}

Ciphertext Scheme::leftRotateFast(Ciphertext& cipher, long rotSlots) {
	uint64_t* bxrot = new uint64_t[cipher.l << context.logN]();
	uint64_t* bx = new uint64_t[cipher.l << context.logN]();

	uint64_t* ax = new uint64_t[(cipher.l + context.K) << context.logN]();

	Key key = leftRotKeyMap.at(rotSlots);

	context.leftRot(bxrot, cipher.bx, cipher.l, rotSlots);
	context.leftRot(bx, cipher.ax, cipher.l, rotSlots);

	context.raiseAndEqual(bx, cipher.l);

	context.mulKey(ax, bx, key.ax, cipher.l);
	context.mulKey(bx, bx, key.bx, cipher.l);

	context.backAndEqual(ax, cipher.l);
	context.backAndEqual(bx, cipher.l);

	context.addAndEqual(bx, bxrot, cipher.l);

	return Ciphertext(ax, bx, context.N, cipher.slots, cipher.l);
}

void Scheme::leftRotateAndEqualFast(Ciphertext& cipher, long rotSlots) {
	uint64_t* bxrot = new uint64_t[cipher.l << context.logN]();
	uint64_t* bx = new uint64_t[cipher.l << context.logN]();
	uint64_t* ax = new uint64_t[(cipher.l + context.K) << context.logN]();

	context.leftRot(bxrot, cipher.bx, cipher.l, rotSlots);
	context.leftRot(bx, cipher.ax, cipher.l, rotSlots);

	Key key = leftRotKeyMap.at(rotSlots);

	context.raiseAndEqual(bx, cipher.l);

	context.mulKey(ax, bx, key.ax, cipher.l);
	context.mulKey(bx, bx, key.bx, cipher.l);

	context.back(cipher.ax, ax, cipher.l);
	context.back(cipher.bx, bx, cipher.l);

	context.addAndEqual(cipher.bx, bxrot, cipher.l);
}

Ciphertext Scheme::leftRotateByPo2(Ciphertext& cipher, long logRotSlots) {
	long rotSlots = (1 << logRotSlots);
	return leftRotateFast(cipher, rotSlots);
}

void Scheme::leftRotateByPo2AndEqual(Ciphertext& cipher, long logRotSlots) {
	long rotSlots = (1 << logRotSlots);
	leftRotateAndEqualFast(cipher, rotSlots);
}

Ciphertext Scheme::rightRotateByPo2(Ciphertext& cipher, long logRotSlots) {
	long rotSlots = context.Nh - (1 << logRotSlots);
	return leftRotateFast(cipher, rotSlots);
}

void Scheme::rightRotateByPo2AndEqual(Ciphertext& cipher, long logRotSlots) {
	long rotSlots = context.Nh - (1 << logRotSlots);
	leftRotateAndEqualFast(cipher, rotSlots);
}

Ciphertext Scheme::leftRotate(Ciphertext& cipher, long rotSlots) {
	Ciphertext res = cipher;
	leftRotateAndEqual(res, rotSlots);
	return res;
}

void Scheme::leftRotateAndEqual(Ciphertext& cipher, long rotSlots) {
	long remrotSlots = rotSlots % cipher.slots;
	long logrotSlots = log2((double)remrotSlots) + 1;
	for (long i = 0; i < logrotSlots; ++i) {
		if(remrotSlots & 1 << i) {
			leftRotateByPo2AndEqual(cipher, i);
		}
	}
}

Ciphertext Scheme::rightRotate(Ciphertext& cipher, long rotSlots) {
	Ciphertext res = cipher;
	rightRotateAndEqual(res, rotSlots);
	return res;
}

void Scheme::rightRotateAndEqual(Ciphertext& cipher, long rotSlots) {
	long remrotSlots = rotSlots % cipher.slots;
	long logrotSlots = log2((double)remrotSlots) + 1;
	for (long i = 0; i < logrotSlots; ++i) {
		if(remrotSlots & 1 << i) {
			rightRotateByPo2AndEqual(cipher, i);
		}
	}
}

Ciphertext Scheme::conjugate(Ciphertext& cipher) {
	uint64_t* bxconj = new uint64_t[context.N * cipher.l];
	uint64_t* bx = new uint64_t[context.N * cipher.l];
	uint64_t* ax = new uint64_t[context.N * (cipher.l + context.K)];

	context.conjugate(bxconj, cipher.bx, cipher.l);
	context.conjugate(bx, cipher.ax, cipher.l);

	Key key = keyMap.at(CONJUGATION);

	context.raiseAndEqual(bx, cipher.l);

	long shift = 0;
	for (long i = 0; i < cipher.l; ++i) {
		context.qiMul(ax + shift, bx + shift, key.ax + shift, i);
		context.qiMulAndEqual(bx + shift, key.bx + shift, i);
		shift += context.N;
	}

	long msshift = context.N * context.L;
	for (long i = 0; i < context.K; ++i) {
		context.piMul(ax + shift, bx + shift, key.ax + msshift, i);
		context.piMulAndEqual(bx + shift, key.bx + msshift, i);
		shift += context.N;
		msshift += context.N;
	}

	context.backAndEqual(ax, cipher.l);
	context.backAndEqual(bx, cipher.l);

	context.addAndEqual(bx, bxconj, cipher.l);

	delete[] bxconj;

	return Ciphertext(ax, bx, context.N, cipher.slots, cipher.l);
}

void Scheme::conjugateAndEqual(Ciphertext& cipher) {
	uint64_t* bxconj = new uint64_t[context.N * cipher.l];
	uint64_t* bx = new uint64_t[context.N * cipher.l];
	uint64_t* ax = new uint64_t[context.N * (cipher.l + context.K)];

	context.conjugate(bxconj, cipher.bx, cipher.l);
	context.conjugate(bx, cipher.ax, cipher.l);

	Key key = keyMap.at(CONJUGATION);

	context.raiseAndEqual(bx, cipher.l);

	long shift = 0;
	for (long i = 0; i < cipher.l; ++i) {
		context.qiMul(ax + shift, bx + shift, key.ax + shift, i);
		context.qiMulAndEqual(bx + shift, key.bx + shift, i);
		shift += context.N;
	}

	long msshift = context.N * context.L;
	for (long i = 0; i < context.K; ++i) {
		context.piMul(ax + shift, bx + shift, key.ax + msshift, i);
		context.piMulAndEqual(bx + shift, key.bx + msshift, i);
		shift += context.N;
		msshift += context.N;
	}

	context.back(cipher.ax, ax, cipher.l);
	context.back(cipher.bx, bx, cipher.l);

	context.addAndEqual(cipher.bx, bxconj, cipher.l);

	delete[] bxconj;
}


