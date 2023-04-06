/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#include "Scheme.h"

#include "NTL/BasicThreadPool.h"
#include "StringUtils.h"
#include "SerializationUtils.h"

Scheme::Scheme(SecretKey& secretKey, Ring& ring, bool isSerialized) : ring(ring), isSerialized(isSerialized) {
	addEncKey(secretKey);
	addMultKey(secretKey);
};

Scheme::~Scheme() {
  for (auto const& t : keyMap)
    delete t.second;
  for (auto const& t : leftRotKeyMap)
    delete t.second;
}

void Scheme::addEncKey(SecretKey& secretKey) {
	ZZ* ax = new ZZ[N];
	ZZ* bx = new ZZ[N];

	long np = ceil((1 + logQQ + logN + 2)/(double)pbnd);
	ring.sampleUniform2(ax, logQQ);
	ring.mult(bx, secretKey.sx, ax, np, QQ);
	ring.subFromGaussAndEqual(bx, QQ);

	Key* key = new Key();
	ring.CRT(key->rax, ax, nprimes);
	ring.CRT(key->rbx, bx, nprimes);
	delete[] ax; delete[] bx;

	if(isSerialized) {
		string path = "serkey/ENCRYPTION.txt";
		SerializationUtils::writeKey(key, path);
		serKeyMap.insert(pair<long, string>(ENCRYPTION, path));
		delete key;
	} else {
		keyMap.insert(pair<long, Key*>(ENCRYPTION, key));
	}
}

void Scheme::addMultKey(SecretKey& secretKey) {
	ZZ* ax = new ZZ[N];
	ZZ* bx = new ZZ[N];
	ZZ* sxsx = new ZZ[N];

	long np = ceil((1 + logQQ + logN + 2)/(double)pbnd);
	ring.sampleUniform2(ax, logQQ);
	ring.mult(bx, secretKey.sx, ax, np, QQ);
	ring.subFromGaussAndEqual(bx, QQ);

	np = ceil((2 + logN + 2)/(double)pbnd);
	ring.mult(sxsx, secretKey.sx, secretKey.sx, np, Q);
	ring.leftShiftAndEqual(sxsx, logQ, QQ);
	ring.addAndEqual(bx, sxsx, QQ);
	delete[] sxsx;

	Key* key = new Key();
	ring.CRT(key->rax, ax, nprimes);
	ring.CRT(key->rbx, bx, nprimes);
	delete[] ax; delete[] bx;
	if(isSerialized) {
		string path = "serkey/MULTIPLICATION.txt";
		SerializationUtils::writeKey(key, path);
		serKeyMap.insert(pair<long, string>(MULTIPLICATION, path));
		delete key;
	} else {
		keyMap.insert(pair<long, Key*>(MULTIPLICATION, key));
	}
}

void Scheme::addConjKey(SecretKey& secretKey) {
	ZZ* ax = new ZZ[N];
	ZZ* bx = new ZZ[N];

	long np = ceil((1 + logQQ + logN + 2)/(double)pbnd);
	ring.sampleUniform2(ax, logQQ);
	ring.mult(bx, secretKey.sx, ax, np, QQ);
	ring.subFromGaussAndEqual(bx, QQ);

	ZZ* sxconj = new ZZ[N];
	ring.conjugate(sxconj, secretKey.sx);
	ring.leftShiftAndEqual(sxconj, logQ, QQ);
	ring.addAndEqual(bx, sxconj, QQ);
	delete[] sxconj;

	Key* key = new Key();
	ring.CRT(key->rax, ax, nprimes);
	ring.CRT(key->rbx, bx, nprimes);
	delete[] ax; delete[] bx;

	if(isSerialized) {
		string path = "serkey/CONJUGATION.txt";
		SerializationUtils::writeKey(key, path);
		serKeyMap.insert(pair<long, string>(CONJUGATION, path));
		delete key;
	} else {
		keyMap.insert(pair<long, Key*>(CONJUGATION, key));
	}
}

void Scheme::addLeftRotKey(SecretKey& secretKey, long r) {
	ZZ* ax = new ZZ[N];
	ZZ* bx = new ZZ[N];

	long np = ceil((1 + logQQ + logN + 2)/(double)pbnd);
	ring.sampleUniform2(ax, logQQ);
	ring.mult(bx, secretKey.sx, ax, np, QQ);
	ring.subFromGaussAndEqual(bx, QQ);

	ZZ* spow = new ZZ[N];
	ring.leftRotate(spow, secretKey.sx, r);
	ring.leftShiftAndEqual(spow, logQ, QQ);
	ring.addAndEqual(bx, spow, QQ);
	delete[] spow;

	Key* key = new Key();
	ring.CRT(key->rax, ax, nprimes);
	ring.CRT(key->rbx, bx, nprimes);
	delete[] ax; delete[] bx;

	if(isSerialized) {
		string path = "serkey/ROTATION_" + to_string(r) + ".txt";
		SerializationUtils::writeKey(key, path);
		serLeftRotKeyMap.insert(pair<long, string>(r, path));
		delete key;
	} else {
		leftRotKeyMap.insert(pair<long, Key*>(r, key));
	}
}

void Scheme::addRightRotKey(SecretKey& secretKey, long r) {
	long idx = Nh - r;
	if(leftRotKeyMap.find(idx) == leftRotKeyMap.end() && serLeftRotKeyMap.find(idx) == serLeftRotKeyMap.end()) {
		addLeftRotKey(secretKey, idx);
	}
}

void Scheme::addLeftRotKeys(SecretKey& secretKey) {
	for (long i = 0; i < logN - 1; ++i) {
		long idx = 1 << i;
		if(leftRotKeyMap.find(idx) == leftRotKeyMap.end() && serLeftRotKeyMap.find(idx) == serLeftRotKeyMap.end()) {
			addLeftRotKey(secretKey, idx);
		}
	}
}

void Scheme::addRightRotKeys(SecretKey& secretKey) {
	for (long i = 0; i < logN - 1; ++i) {
		long idx = Nh - (1 << i);
		if(leftRotKeyMap.find(idx) == leftRotKeyMap.end() && serLeftRotKeyMap.find(idx) == serLeftRotKeyMap.end()) {
			addLeftRotKey(secretKey, idx);
		}
	}
}

void Scheme::addBootKey(SecretKey& secretKey, long logl, long logp) {
	ring.addBootContext(logl, logp);

	addConjKey(secretKey);
	addLeftRotKeys(secretKey);

	long loglh = logl/2;
	long k = 1 << loglh;
	long m = 1 << (logl - loglh);

	for (long i = 1; i < k; ++i) {
		if(leftRotKeyMap.find(i) == leftRotKeyMap.end() && serLeftRotKeyMap.find(i) == serLeftRotKeyMap.end()) {
			addLeftRotKey(secretKey, i);
		}
	}

	for (long i = 1; i < m; ++i) {
		long idx = i * k;
		if(leftRotKeyMap.find(idx) == leftRotKeyMap.end() && serLeftRotKeyMap.find(idx) == serLeftRotKeyMap.end()) {
			addLeftRotKey(secretKey, idx);
		}
	}
}

void Scheme::encode(Plaintext& plain, double* vals, long n, long logp, long logq) {
	plain.logp = logp;
	plain.logq = logq;
	plain.n = n;
	ring.encode(plain.mx, vals, n, logp + logQ);
}

void Scheme::encode(Plaintext& plain, complex<double>* vals, long n, long logp, long logq) {
	plain.logp = logp;
	plain.logq = logq;
	plain.n = n;
	ring.encode(plain.mx, vals, n, logp + logQ);
}

complex<double>* Scheme::decode(Plaintext& plain) {
	complex<double>* res = new complex<double>[plain.n];
	ring.decode(plain.mx, res, plain.n, plain.logp, plain.logq);
	return res;
}

void Scheme::encodeSingle(Plaintext& plain, double val, long logp, long logq) {
	plain.logp = logp;
	plain.logq = logq;
	plain.n = 1;
	plain.mx[0] = EvaluatorUtils::scaleUpToZZ(val, logp + logQ);
}

void Scheme::encodeSingle(Plaintext& plain, complex<double> val, long logp, long logq) {
	plain.logp = logp;
	plain.logq = logq;
	plain.n = 1;
	plain.mx[0] = EvaluatorUtils::scaleUpToZZ(val.real(), logp + logQ);
	plain.mx[Nh] = EvaluatorUtils::scaleUpToZZ(val.imag(), logp + logQ);
}

complex<double> Scheme::decodeSingle(Plaintext& plain) {
	ZZ q = ring.qpows[plain.logq];

	complex<double> res;
	ZZ tmp = plain.mx[0] % q;
	if(NumBits(tmp) == plain.logq) tmp -= q;
	res.real(EvaluatorUtils::scaleDownToReal(tmp, plain.logp));

	tmp = plain.mx[Nh] % q;
	if(NumBits(tmp) == plain.logq) tmp -= q;
	res.imag(EvaluatorUtils::scaleDownToReal(tmp, plain.logp));

	return res;
}

void Scheme::encryptMsg(Ciphertext& cipher, Plaintext& plain) {
	cipher.logp = plain.logp;
	cipher.logq = plain.logq;
	cipher.n = plain.n;
	ZZ qQ = ring.qpows[plain.logq + logQ];

	ZZ* vx = new ZZ[N];
	ring.sampleZO(vx);

	Key* key = isSerialized ? SerializationUtils::readKey(serKeyMap.at(ENCRYPTION)) : keyMap.at(ENCRYPTION);

	long np = ceil((1 + logQQ + logN + 2)/(double)pbnd);
	ring.multNTT(cipher.ax, vx, key->rax, np, qQ);
	ring.addGaussAndEqual(cipher.ax, qQ);

	ring.multNTT(cipher.bx, vx, key->rbx, np, qQ);
	ring.addGaussAndEqual(cipher.bx, qQ);
	delete[] vx;

	ring.addAndEqual(cipher.bx, plain.mx, qQ);

	ring.rightShiftAndEqual(cipher.ax, logQ);
	ring.rightShiftAndEqual(cipher.bx, logQ);
}

void Scheme::decryptMsg(Plaintext& plain, SecretKey& secretKey, Ciphertext& cipher) {
	ZZ q = ring.qpows[cipher.logq];
	plain.logp = cipher.logp;
	plain.logq = cipher.logq;
	plain.n = cipher.n;
	long np = ceil((1 + cipher.logq + logN + 2)/(double)pbnd);
	ring.mult(plain.mx, cipher.ax, secretKey.sx, np, q);
	ring.addAndEqual(plain.mx, cipher.bx, q);
}

void Scheme::encrypt(Ciphertext& cipher, complex<double>* vals, long n, long logp, long logq) {
	Plaintext plain;
	encode(plain, vals, n, logp, logq);
	encryptMsg(cipher, plain);
}

void Scheme::encrypt(Ciphertext& cipher, double* vals, long n, long logp, long logq) {
	Plaintext plain;
	encode(plain, vals, n, logp, logq);
	encryptMsg(cipher, plain);
}

void Scheme::encryptBySk(Ciphertext& cipher, SecretKey& secretKey, complex<double>* vals, long n, long logp, long logq, double sigma1) {
	Plaintext plain;
	encode(plain, vals, n, logp, logq);

	cipher.logp = plain.logp;
	cipher.logq = plain.logq;
	cipher.n = plain.n;

	long np = ceil((1 + logQQ + logN + 2)/(double)pbnd);
	ring.sampleUniform2(cipher.ax, logQQ);
	ring.mult(cipher.bx, secretKey.sx, cipher.ax, np, QQ);
	ring.subFromGaussAndEqual(cipher.bx, QQ, sigma1);
	ring.addAndEqual(cipher.bx, plain.mx, QQ);

	ring.rightShiftAndEqual(cipher.ax, logQ);
	ring.rightShiftAndEqual(cipher.bx, logQ);
}

void Scheme::encryptBySk(Ciphertext& cipher, SecretKey& secretKey, double* vals, long n, long logp, long logq, double sigma1) {
	Plaintext plain;
	encode(plain, vals, n, logp, logq);

	cipher.logp = plain.logp;
	cipher.logq = plain.logq;
	cipher.n = plain.n;

	long np = ceil((1 + logQQ + logN + 2)/(double)pbnd);
	ring.sampleUniform2(cipher.ax, logQQ);
	ring.mult(cipher.bx, secretKey.sx, cipher.ax, np, QQ);
	ring.subFromGaussAndEqual(cipher.bx, QQ, sigma1);
	ring.addAndEqual(cipher.bx, plain.mx, QQ);

	ring.rightShiftAndEqual(cipher.ax, logQ);
	ring.rightShiftAndEqual(cipher.bx, logQ);
}

void Scheme::encryptZeros(Ciphertext& cipher, long n, long logp, long logq) {
	encryptSingle(cipher, 0.0, logp, logq);
	cipher.n = n;
}

complex<double>* Scheme::decrypt(SecretKey& secretKey, Ciphertext& cipher) {
	Plaintext plain;
	decryptMsg(plain, secretKey, cipher);
	return decode(plain);
}

complex<double>* Scheme::decryptForShare(SecretKey& secretKey, Ciphertext& cipher, long logErrBound) {
	Plaintext plain;
	decryptMsg(plain, secretKey, cipher);
	
	double sigma1 = sigma * sqrt(2);
	double sigma2;
	double PI = 4.0 * atan(1.0);
	if (logErrBound == -1) {
		// WIDTH = 8, WIDTH1 = sqrt(2*pi)*sigma
		// Note that sigma is already multiplied by sqrt(2) in TestScheme.cpp
		double WIDTH_sq = 64.0; 
		double WIDTH1_sq = 2.0*PI*sigma1*sigma1;
		sigma2 = sqrt(WIDTH1_sq * WIDTH_sq / ((WIDTH1_sq - WIDTH_sq) * 2 * PI));
	} else {
		sigma2 = ((double) ((long) 1 << logErrBound)) / sqrt(2 * PI);
	}
	ZZ q = ring.qpows[plain.logq];
	ring.addGaussAndEqual(plain.mx, q, sigma2);
	
	return decode(plain);
}

void Scheme::encryptSingle(Ciphertext& cipher, complex<double> val, long logp, long logq) {
	Plaintext plain;
	encodeSingle(plain, val, logp, logq);
	encryptMsg(cipher, plain);
}

void Scheme::encryptSingle(Ciphertext& cipher, double val, long logp, long logq) {
	Plaintext plain;
	encodeSingle(plain, val, logp, logq);
	encryptMsg(cipher, plain);
}

complex<double> Scheme::decryptSingle(SecretKey& secretKey, Ciphertext& cipher) {
	Plaintext plain;
	decryptMsg(plain, secretKey, cipher);
	return decodeSingle(plain);
}

//-----------------------------------------

void Scheme::negate(Ciphertext& res, Ciphertext& cipher) {
	res.copyParams(cipher);
	ring.negate(res.ax, cipher.ax);
	ring.negate(res.bx, cipher.bx);
}

void Scheme::negateAndEqual(Ciphertext& cipher) {
	ring.negateAndEqual(cipher.ax);
	ring.negateAndEqual(cipher.bx);
}

void Scheme::add(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2) {
	ZZ q = ring.qpows[cipher1.logq];
	res.copyParams(cipher1);
	ring.add(res.ax, cipher1.ax, cipher2.ax, q);
	ring.add(res.bx, cipher1.bx, cipher2.bx, q);
}

void Scheme::addAndEqual(Ciphertext& cipher1, Ciphertext& cipher2) {
	ZZ q = ring.qpows[cipher1.logq];
	ring.addAndEqual(cipher1.ax, cipher2.ax, q);
	ring.addAndEqual(cipher1.bx, cipher2.bx, q);
}

//-----------------------------------------

void Scheme::addConst(Ciphertext& res, Ciphertext& cipher, double cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstZZ = logp < 0 ? EvaluatorUtils::scaleUpToZZ(cnst, cipher.logp) : EvaluatorUtils::scaleUpToZZ(cnst, logp);
	res.copy(cipher);
	AddMod(res.bx[0], res.bx[0], cnstZZ, q);
}

void Scheme::addConst(Ciphertext& res, Ciphertext& cipher, RR& cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstZZ = logp < 0 ? EvaluatorUtils::scaleUpToZZ(cnst, cipher.logp) : EvaluatorUtils::scaleUpToZZ(cnst, logp);
	res.copy(cipher);
	AddMod(res.bx[0], res.bx[0], cnstZZ, q);
}

void Scheme::addConst(Ciphertext& res, Ciphertext& cipher, complex<double> cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstZZ = logp < 0 ? EvaluatorUtils::scaleUpToZZ(cnst.real(), cipher.logp) : EvaluatorUtils::scaleUpToZZ(cnst.real(), logp);
	res.copy(cipher);
	AddMod(res.bx[0], cipher.bx[0], cnstZZ, q);
}

void Scheme::addConstAndEqual(Ciphertext& cipher, double cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstZZ = logp < 0 ? EvaluatorUtils::scaleUpToZZ(cnst, cipher.logp) : EvaluatorUtils::scaleUpToZZ(cnst, logp);
	AddMod(cipher.bx[0], cipher.bx[0], cnstZZ, q);
}

void Scheme::addConstAndEqual(Ciphertext& cipher, RR& cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstZZ = logp < 0 ? EvaluatorUtils::scaleUpToZZ(cnst, cipher.logp) : EvaluatorUtils::scaleUpToZZ(cnst, logp);
	AddMod(cipher.bx[0], cipher.bx[0], cnstZZ, q);
}

void Scheme::addConstAndEqual(Ciphertext& cipher, complex<double> cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstrZZ = logp < 0 ? EvaluatorUtils::scaleUpToZZ(cnst.real(), cipher.logp) : EvaluatorUtils::scaleUpToZZ(cnst.real(), logp);
	ZZ cnstiZZ = logp < 0 ? EvaluatorUtils::scaleUpToZZ(cnst.imag(), cipher.logp) : EvaluatorUtils::scaleUpToZZ(cnst.imag(), logp);
	AddMod(cipher.bx[0], cipher.bx[0], cnstrZZ, q);
	AddMod(cipher.bx[Nh], cipher.bx[Nh], cnstiZZ, q);
}

//-----------------------------------------

void Scheme::sub(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2) {
	ZZ q = ring.qpows[cipher1.logq];
	res.copyParams(cipher1);
	ring.sub(res.ax, cipher1.ax, cipher2.ax, q);
	ring.sub(res.bx, cipher1.bx, cipher2.bx, q);
}

void Scheme::subAndEqual(Ciphertext& cipher1, Ciphertext& cipher2) {
	ZZ q = ring.qpows[cipher1.logq];
	ring.subAndEqual(cipher1.ax, cipher2.ax, q);
	ring.subAndEqual(cipher1.bx, cipher2.bx, q);
}

void Scheme::subAndEqual2(Ciphertext& cipher1, Ciphertext& cipher2) {
	ZZ q = ring.qpows[cipher1.logq];
	ring.subAndEqual2(cipher1.ax, cipher2.ax, q);
	ring.subAndEqual2(cipher1.bx, cipher2.bx, q);
}

void Scheme::imult(Ciphertext& res, Ciphertext& cipher) {
	ZZ q = ring.qpows[cipher.logq];
	res.copyParams(cipher);
	ring.multByMonomial(res.ax, cipher.ax, Nh);
	ring.multByMonomial(res.bx, cipher.bx, Nh);
}

void Scheme::idiv(Ciphertext& res, Ciphertext& cipher) {
	ZZ q = ring.qpows[cipher.logq];
	res.copyParams(cipher);
	ring.multByMonomial(res.ax, cipher.ax, 3 * Nh);
	ring.multByMonomial(res.bx, cipher.bx, 3 * Nh);
}

void Scheme::imultAndEqual(Ciphertext& cipher) {
	ring.multByMonomialAndEqual(cipher.ax, Nh);
	ring.multByMonomialAndEqual(cipher.bx, Nh);
}

void Scheme::idivAndEqual(Ciphertext& cipher) {
	ring.multByMonomialAndEqual(cipher.ax, 3 * Nh);
	ring.multByMonomialAndEqual(cipher.bx, 3 * Nh);
}

void Scheme::mult(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2) {
	res.copyParams(cipher1);
	res.logp += cipher2.logp;

	ZZ q = ring.qpows[cipher1.logq];
	ZZ qQ = ring.qpows[cipher1.logq + logQ];

	long np = ceil((2 + cipher1.logq + cipher2.logq + logN + 2)/(double)pbnd);

	uint64_t* ra1 = new uint64_t[np << logN];
	uint64_t* rb1 = new uint64_t[np << logN];
	uint64_t* ra2 = new uint64_t[np << logN];
	uint64_t* rb2 = new uint64_t[np << logN];

	ring.CRT(ra1, cipher1.ax, np);
	ring.CRT(rb1, cipher1.bx, np);
	ring.CRT(ra2, cipher2.ax, np);
	ring.CRT(rb2, cipher2.bx, np);

	ZZ* axax = new ZZ[N];
	ZZ* bxbx = new ZZ[N];
	ZZ* axbx = new ZZ[N];
	ring.multDNTT(axax, ra1, ra2, np, q);
	ring.multDNTT(bxbx, rb1, rb2, np, q);

	ring.addNTTAndEqual(ra1, rb1, np);
	ring.addNTTAndEqual(ra2, rb2, np);
	ring.multDNTT(axbx, ra1, ra2, np, q);

	Key* key = isSerialized ? SerializationUtils::readKey(serKeyMap.at(MULTIPLICATION)) : keyMap.at(MULTIPLICATION);

	np = ceil((cipher1.logq + logQQ + logN + 2)/(double)pbnd);
	uint64_t* raa = new uint64_t[np << logN];
	ring.CRT(raa, axax, np);
	ring.multDNTT(res.ax, raa, key->rax, np, qQ);
	ring.multDNTT(res.bx, raa, key->rbx, np, qQ);
	ring.rightShiftAndEqual(res.ax, logQ);
	ring.rightShiftAndEqual(res.bx, logQ);

	ring.addAndEqual(res.ax, axbx, q);
	ring.subAndEqual(res.ax, bxbx, q);
	ring.subAndEqual(res.ax, axax, q);
	ring.addAndEqual(res.bx, bxbx, q);

	delete[] axax;
	delete[] bxbx;
	delete[] axbx;
	delete[] ra1;
	delete[] ra2;
	delete[] rb1;
	delete[] rb2;
	delete[] raa;
}

void Scheme::multAndEqual(Ciphertext& cipher1, Ciphertext& cipher2) {

	ZZ q = ring.qpows[cipher1.logq];
	ZZ qQ = ring.qpows[cipher1.logq + logQ];

	long np = ceil((2 + cipher1.logq + cipher2.logq + logN + 2)/(double)pbnd);

	uint64_t* ra1 = new uint64_t[np << logN];
	uint64_t* rb1 = new uint64_t[np << logN];
	uint64_t* ra2 = new uint64_t[np << logN];
	uint64_t* rb2 = new uint64_t[np << logN];

	ring.CRT(ra1, cipher1.ax, np);
	ring.CRT(rb1, cipher1.bx, np);
	ring.CRT(ra2, cipher2.ax, np);
	ring.CRT(rb2, cipher2.bx, np);

	ZZ* axax = new ZZ[N];
	ZZ* bxbx = new ZZ[N];
	ZZ* axbx = new ZZ[N];

	ring.multDNTT(axax, ra1, ra2, np, q);
	ring.multDNTT(bxbx, rb1, rb2, np, q);
	ring.addNTTAndEqual(ra1, rb1, np);
	ring.addNTTAndEqual(ra2, rb2, np);
	ring.multDNTT(axbx, ra1, ra2, np, q);

	Key* key = isSerialized ? SerializationUtils::readKey(serKeyMap.at(MULTIPLICATION)) : keyMap.at(MULTIPLICATION);

	np = ceil((cipher1.logq + logQQ + logN + 2)/(double)pbnd);
	uint64_t* raa = new uint64_t[np << logN];
	ring.CRT(raa, axax, np);
	ring.multDNTT(cipher1.ax, raa, key->rax, np, qQ);
	ring.multDNTT(cipher1.bx, raa, key->rbx, np, qQ);

	ring.rightShiftAndEqual(cipher1.ax, logQ);
	ring.rightShiftAndEqual(cipher1.bx, logQ);

	ring.addAndEqual(cipher1.ax, axbx, q);
	ring.subAndEqual(cipher1.ax, bxbx, q);
	ring.subAndEqual(cipher1.ax, axax, q);
	ring.addAndEqual(cipher1.bx, bxbx, q);

	delete[] axax;
	delete[] bxbx;
	delete[] axbx;
	delete[] ra1;
	delete[] ra2;
	delete[] rb1;
	delete[] rb2;
	delete[] raa;

	cipher1.logp += cipher2.logp;
}

//-----------------------------------------

void Scheme::square(Ciphertext& res, Ciphertext& cipher) {
	res.copyParams(cipher);
	res.logp += cipher.logp;
	ZZ q = ring.qpows[cipher.logq];
	ZZ qQ = ring.qpows[cipher.logq + logQ];

	long np = ceil((2 * cipher.logq + logN + 2)/(double)pbnd);

	uint64_t* ra = new uint64_t[np << logN];
	uint64_t* rb = new uint64_t[np << logN];

	ring.CRT(ra, cipher.ax, np);
	ring.CRT(rb, cipher.bx, np);

	ZZ* axax = new ZZ[N];
	ZZ* axbx = new ZZ[N];
	ZZ* bxbx = new ZZ[N];

	ring.squareNTT(bxbx, rb, np, q);
	ring.squareNTT(axax, ra, np, q);
	ring.multDNTT(axbx, ra, rb, np, q);
	ring.addAndEqual(axbx, axbx, q);

	Key* key = isSerialized ? SerializationUtils::readKey(serKeyMap.at(MULTIPLICATION)) : keyMap.at(MULTIPLICATION);

	np = ceil((cipher.logq + logQQ + logN + 2)/(double)pbnd);
	uint64_t* raa = new uint64_t[np << logN];
	ring.CRT(raa, axax, np);
	ring.multDNTT(res.ax, raa, key->rax, np, qQ);
	ring.multDNTT(res.bx, raa, key->rbx, np, qQ);

	ring.rightShiftAndEqual(res.ax, logQ);
	ring.rightShiftAndEqual(res.bx, logQ);

	ring.addAndEqual(res.ax, axbx, q);
	ring.addAndEqual(res.bx, bxbx, q);

	delete[] axbx;
	delete[] axax;
	delete[] bxbx;

	delete[] ra;
	delete[] rb;
	delete[] raa;
}

void Scheme::squareAndEqual(Ciphertext& cipher) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ qQ = ring.qpows[cipher.logq + logQ];

	long np = ceil((2 + 2 * cipher.logq + logN + 2)/(double)pbnd);

	uint64_t* ra = new uint64_t[np << logN];
	uint64_t* rb = new uint64_t[np << logN];

	ring.CRT(ra, cipher.ax, np);
	ring.CRT(rb, cipher.bx, np);

	ZZ* axax = new ZZ[N];
	ZZ* axbx = new ZZ[N];
	ZZ* bxbx = new ZZ[N];

	ring.squareNTT(bxbx, rb, np, q);
	ring.squareNTT(axax, ra, np, q);

	ring.multDNTT(axbx, ra, rb, np, q);
	ring.addAndEqual(axbx, axbx, q);

	Key* key = isSerialized ? SerializationUtils::readKey(serKeyMap.at(MULTIPLICATION)) : keyMap.at(MULTIPLICATION);

	np = ceil((cipher.logq + logQQ + logN + 2)/(double)pbnd);

	uint64_t* raa = new uint64_t[np << logN];
	ring.CRT(raa, axax, np);
	ring.multDNTT(cipher.ax, raa, key->rax, np, qQ);
	ring.multDNTT(cipher.bx, raa, key->rbx, np, qQ);

	ring.rightShiftAndEqual(cipher.ax, logQ);
	ring.rightShiftAndEqual(cipher.bx, logQ);

	ring.addAndEqual(cipher.ax, axbx, q);
	ring.addAndEqual(cipher.bx, bxbx, q);
	cipher.logp *= 2;

	delete[] axbx;
	delete[] axax;
	delete[] bxbx;

	delete[] ra;
	delete[] rb;
	delete[] raa;
}

//-----------------------------------------

void Scheme::multByConst(Ciphertext& res, Ciphertext& cipher, double cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstZZ = EvaluatorUtils::scaleUpToZZ(cnst, logp);
	ring.multByConst(res.ax, cipher.ax, cnstZZ, q);
	ring.multByConst(res.bx, cipher.bx, cnstZZ, q);
	res.copyParams(cipher);
	res.logp += logp;
}

void Scheme::multByConst(Ciphertext& res, Ciphertext& cipher, complex<double> cnst, long logp) {
	res.copy(cipher);
	multByConstAndEqual(res, cnst, logp);
}

void Scheme::multByConstVec(Ciphertext& res, Ciphertext& cipher, complex<double>* cnstVec, long logp) {
	res.copy(cipher);
	multByConstVecAndEqual(res, cnstVec, logp);
}

void Scheme::multByConstVecAndEqual(Ciphertext& cipher, complex<double>* cnstVec, long logp) {
	long slots = cipher.n;
	ZZ* cnstPoly = new ZZ[N];
	ring.encode(cnstPoly, cnstVec, slots, logp);
	multByPolyAndEqual(cipher, cnstPoly, logp);
	delete[] cnstPoly;
}

void Scheme::multByConstAndEqual(Ciphertext& cipher, double cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstZZ = EvaluatorUtils::scaleUpToZZ(cnst, logp);
	ring.multByConstAndEqual(cipher.ax, cnstZZ, q);
	ring.multByConstAndEqual(cipher.bx, cnstZZ, q);
	cipher.logp += logp;
}

void Scheme::multByConstAndEqual(Ciphertext& cipher, RR& cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstZZ = EvaluatorUtils::scaleUpToZZ(cnst, logp);
	ring.multByConstAndEqual(cipher.ax, cnstZZ, q);
	ring.multByConstAndEqual(cipher.bx, cnstZZ, q);
	cipher.logp += logp;
}

void Scheme::multByConstAndEqual(Ciphertext& cipher, complex<double> cnst, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ cnstZZReal = EvaluatorUtils::scaleUpToZZ(cnst.real(), logp);
	ZZ cnstZZImag = EvaluatorUtils::scaleUpToZZ(cnst.imag(), logp);

	Ciphertext tmp; // compute imagnary part
	tmp.copyParams(cipher);
	ring.multByConst(tmp.ax, cipher.ax, cnstZZImag, q);
	ring.multByConst(tmp.bx, cipher.bx, cnstZZImag, q);
	ring.multByMonomialAndEqual(tmp.ax, N / 2);
	ring.multByMonomialAndEqual(tmp.bx, N / 2);
	

	ring.multByConstAndEqual(cipher.ax, cnstZZReal, q);
	ring.multByConstAndEqual(cipher.bx, cnstZZReal, q);

	ring.addAndEqual(cipher.ax, tmp.ax, QQ);
	ring.addAndEqual(cipher.bx, tmp.bx, QQ);

	cipher.logp += logp;
}

void Scheme::multByPoly(Ciphertext& res, Ciphertext& cipher, ZZ* poly, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	res.copyParams(cipher);
	long bnd = ring.maxBits(poly, N);
	long np = ceil((cipher.logq + bnd + logN + 2)/(double)pbnd);
	uint64_t* rpoly = new uint64_t[np << logN];
	ring.CRT(rpoly, poly, np);
	ring.multNTT(res.ax, cipher.ax, rpoly, np, q);
	ring.multNTT(res.bx, cipher.bx, rpoly, np, q);
	delete[] rpoly;
	res.logp += logp;
}

void Scheme::multByPolyAndEqual(Ciphertext& cipher, ZZ* poly, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	long bnd = ring.maxBits(poly, N);
	long np = ceil((cipher.logq + bnd + logN + 2)/(double)pbnd);
	uint64_t* rpoly = new uint64_t[np << logN];
	ring.CRT(rpoly, poly, np);
	ring.multNTTAndEqual(cipher.ax, rpoly, np, q);
	ring.multNTTAndEqual(cipher.bx, rpoly, np, q);
	delete[] rpoly;
	cipher.logp += logp;
}

void Scheme::multByPolyNTT(Ciphertext& res, Ciphertext& cipher, uint64_t* rpoly, long bnd, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	res.copyParams(cipher);
	long np = ceil((cipher.logq + bnd + logN + 2)/(double)pbnd);
	ring.multNTT(res.ax, cipher.ax, rpoly, np, q);
	ring.multNTT(res.bx, cipher.bx, rpoly, np, q);
	res.logp += logp;
}

void Scheme::multByPolyNTTAndEqual(Ciphertext& cipher, uint64_t* rpoly, long bnd, long logp) {
	ZZ q = ring.qpows[cipher.logq];
	long np = ceil((cipher.logq + bnd + logN + 2)/(double)pbnd);
	ring.multNTTAndEqual(cipher.ax, rpoly, np, q);
	ring.multNTTAndEqual(cipher.bx, rpoly, np, q);
	cipher.logp += logp;
}

//-----------------------------------------

void Scheme::multByMonomial(Ciphertext& res, Ciphertext& cipher, const long degree) {
	res.copyParams(cipher);
	ring.multByMonomial(res.ax, cipher.ax, degree);
	ring.multByMonomial(res.bx, cipher.bx, degree);
}

void Scheme::multByMonomialAndEqual(Ciphertext& cipher, const long degree) {
	ring.multByMonomialAndEqual(cipher.ax, degree);
	ring.multByMonomialAndEqual(cipher.bx, degree);
}

//-----------------------------------------

void Scheme::leftShift(Ciphertext& res, Ciphertext& cipher, long bits) {
	ZZ q = ring.qpows[cipher.logq];
	res.copyParams(cipher);
	ring.leftShift(res.ax, cipher.ax, bits, q);
	ring.leftShift(res.bx, cipher.bx, bits, q);
}

void Scheme::leftShiftAndEqual(Ciphertext& cipher, long bits) {
	ZZ q = ring.qpows[cipher.logq];
	ring.leftShiftAndEqual(cipher.ax, bits, q);
	ring.leftShiftAndEqual(cipher.bx, bits, q);
}

void Scheme::doubleAndEqual(Ciphertext& cipher) {
	ZZ q = ring.qpows[cipher.logq];
	ring.doubleAndEqual(cipher.ax, q);
	ring.doubleAndEqual(cipher.bx, q);
}

void Scheme::divByPo2(Ciphertext& res, Ciphertext& cipher, long bits) {
	res.copyParams(cipher);
	ring.rightShift(res.ax, cipher.ax, bits);
	ring.rightShift(res.bx, cipher.bx, bits);
	res.logq -= bits;
}

void Scheme::divByPo2AndEqual(Ciphertext& cipher, long bits) {
	ring.rightShiftAndEqual(cipher.ax, bits);
	ring.rightShiftAndEqual(cipher.bx, bits);
	cipher.logq -= bits;
}


//-----------------------------------------

void Scheme::reScaleBy(Ciphertext& res, Ciphertext& cipher, long dlogq) {
	res.copyParams(cipher);
	ring.rightShift(res.ax, cipher.ax, dlogq);
	ring.rightShift(res.bx, cipher.bx, dlogq);
	res.logp -= dlogq;
	res.logq -= dlogq;
}

void Scheme::reScaleTo(Ciphertext& res, Ciphertext& cipher, long logq) {
	long dlogq = cipher.logq - logq;
	res.copyParams(cipher);
	ring.rightShift(res.ax, cipher.ax, dlogq);
	ring.rightShift(res.bx, cipher.bx, dlogq);
	res.logp -= dlogq;
}

void Scheme::reScaleByAndEqual(Ciphertext& cipher, long dlogq) {
	ring.rightShiftAndEqual(cipher.ax, dlogq);
	ring.rightShiftAndEqual(cipher.bx, dlogq);
	cipher.logq -= dlogq;
	cipher.logp -= dlogq;
}

void Scheme::reScaleToAndEqual(Ciphertext& cipher, long logq) {
	long dlogq = cipher.logq - logq;
	ring.rightShiftAndEqual(cipher.ax, dlogq);
	ring.rightShiftAndEqual(cipher.bx, dlogq);
	cipher.logq = logq;
	cipher.logp -= dlogq;
}

void Scheme::modDownBy(Ciphertext& res, Ciphertext& cipher, long dlogq) {
	ZZ q = ring.qpows[cipher.logq - dlogq];
	res.copyParams(cipher);
	ring.mod(res.ax, cipher.ax, q);
	ring.mod(res.bx, cipher.bx, q);
	res.logq -= dlogq;
}

void Scheme::modDownByAndEqual(Ciphertext& cipher, long dlogq) {
	ZZ q = ring.qpows[cipher.logq - dlogq];
	ring.modAndEqual(cipher.ax, q);
	ring.modAndEqual(cipher.bx, q);
	cipher.logq -= dlogq;
}

void Scheme::modDownTo(Ciphertext& res, Ciphertext& cipher, long logq) {
	ZZ q = ring.qpows[logq];
	res.copyParams(cipher);
	ring.mod(res.ax, cipher.ax, q);
	ring.mod(res.bx, cipher.bx, q);
	res.logq = logq;
}

void Scheme::modDownToAndEqual(Ciphertext& cipher, long logq) {
	ZZ q = ring.qpows[logq];
	cipher.logq = logq;
	ring.modAndEqual(cipher.ax, q);
	ring.modAndEqual(cipher.bx, q);
}


//----------------------------------------------------------------------------------
//   ROTATIONS & CONJUGATIONS
//----------------------------------------------------------------------------------


void Scheme::leftRotateFast(Ciphertext& res, Ciphertext& cipher, long r) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ qQ = ring.qpows[cipher.logq + logQ];

	ZZ* bxrot = new ZZ[N];
	ZZ* axrot = new ZZ[N];

	ring.leftRotate(bxrot, cipher.bx, r);
	ring.leftRotate(axrot, cipher.ax, r);

	Key* key = isSerialized ? SerializationUtils::readKey(serLeftRotKeyMap.at(r)) : leftRotKeyMap.at(r);
	res.copyParams(cipher);

	long np = ceil((cipher.logq + logQQ + logN + 2)/(double)pbnd);
	uint64_t* rarot = new uint64_t[np << logN];
	ring.CRT(rarot, axrot, np);
	ring.multDNTT(res.ax, rarot, key->rax, np, qQ);
	ring.multDNTT(res.bx, rarot, key->rbx, np, qQ);

	ring.rightShiftAndEqual(res.ax, logQ);
	ring.rightShiftAndEqual(res.bx, logQ);
	ring.addAndEqual(res.bx, bxrot, q);
	delete[] bxrot;
	delete[] axrot;
	delete[] rarot;
}

void Scheme::leftRotateFastAndEqual(Ciphertext& cipher, long r) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ qQ = ring.qpows[cipher.logq + logQ];

	ZZ* bxrot = new ZZ[N];
	ZZ* axrot = new ZZ[N];

	ring.leftRotate(bxrot, cipher.bx, r);
	ring.leftRotate(axrot, cipher.ax, r);
	Key* key = isSerialized ? SerializationUtils::readKey(serLeftRotKeyMap.at(r)) : leftRotKeyMap.at(r);
	long np = ceil((cipher.logq + logQQ + logN + 2)/(double)pbnd);
	uint64_t* rarot = new uint64_t[np << logN];
	ring.CRT(rarot, axrot, np);
	ring.multDNTT(cipher.ax, rarot, key->rax, np, qQ);
	ring.multDNTT(cipher.bx, rarot, key->rbx, np, qQ);

	ring.rightShiftAndEqual(cipher.ax, logQ);
	ring.rightShiftAndEqual(cipher.bx, logQ);

	ring.addAndEqual(cipher.bx, bxrot, q);

	delete[] bxrot;
	delete[] axrot;
	delete[] rarot;
}

void Scheme::rightRotateFast(Ciphertext& res, Ciphertext& cipher, long r) {
	long rr = Nh - r;
	leftRotateFast(res, cipher, rr);
}

void Scheme::rightRotateFastAndEqual(Ciphertext& cipher, long r) {
	long rr = Nh - r;
	leftRotateFastAndEqual(cipher, rr);
}

void Scheme::conjugate(Ciphertext& res, Ciphertext& cipher) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ qQ = ring.qpows[cipher.logq + logQ];

	ZZ* bxconj = new ZZ[N];
	ZZ* axconj = new ZZ[N];

	ring.conjugate(bxconj, cipher.bx);
	ring.conjugate(axconj, cipher.ax);

	Key* key = isSerialized ? SerializationUtils::readKey(serKeyMap.at(CONJUGATION)) : keyMap.at(CONJUGATION);
	res.copyParams(cipher);
	long np = ceil((cipher.logq + logQQ + logN + 2)/(double)pbnd);
	uint64_t* raconj = new uint64_t[np << logN];
	ring.CRT(raconj, axconj, np);
	ring.multDNTT(res.ax, raconj, key->rax, np, qQ);
	ring.multDNTT(res.bx, raconj, key->rbx, np, qQ);

	ring.rightShiftAndEqual(res.ax, logQ);
	ring.rightShiftAndEqual(res.bx, logQ);
	ring.addAndEqual(res.bx, bxconj, q);

	delete[] bxconj;
	delete[] axconj;
	delete[] raconj;
}

void Scheme::conjugateAndEqual(Ciphertext& cipher) {
	ZZ q = ring.qpows[cipher.logq];
	ZZ qQ = ring.qpows[cipher.logq + logQ];

	ZZ* bxconj = new ZZ[N];
	ZZ* axconj = new ZZ[N];

	ring.conjugate(bxconj, cipher.bx);
	ring.conjugate(axconj, cipher.ax);

	Key* key = isSerialized ? SerializationUtils::readKey(serKeyMap.at(CONJUGATION)) : keyMap.at(CONJUGATION);

	long np = ceil((cipher.logq + logQQ + logN + 2)/(double)pbnd);
	uint64_t* raconj = new uint64_t[np << logN];
	ring.CRT(raconj, axconj, np);
	ring.multDNTT(cipher.ax, raconj, key->rax, np, qQ);
	ring.multDNTT(cipher.bx, raconj, key->rbx, np, qQ);

	ring.rightShiftAndEqual(cipher.ax, logQ);
	ring.rightShiftAndEqual(cipher.bx, logQ);

	ring.addAndEqual(cipher.bx, bxconj, q);

	delete[] bxconj;
	delete[] axconj;
	delete[] raconj;
}


//----------------------------------------------------------------------------------
//   BOOTSTRAPPING
//----------------------------------------------------------------------------------


void Scheme::normalizeAndEqual(Ciphertext& cipher) {
	ZZ q = ring.qpows[cipher.logq];

	for (long i = 0; i < N; ++i) {
		if(NumBits(cipher.ax[i]) == cipher.logq) cipher.ax[i] -= q;
		if(NumBits(cipher.bx[i]) == cipher.logq) cipher.bx[i] -= q;
	}
}

void Scheme::coeffToSlotAndEqual(Ciphertext& cipher) {
	long slots = cipher.n;
	long logSlots = log2(slots);
	long logk = logSlots / 2;
	long k = 1 << logk;

	Ciphertext* rotvec = new Ciphertext[k];
	rotvec[0].copy(cipher);

	NTL_EXEC_RANGE(k - 1, first, last);
	for (long j = first; j < last; ++j) {
		leftRotateFast(rotvec[j+1], rotvec[0], j + 1);
	}
	NTL_EXEC_RANGE_END;

	BootContext* bootContext = ring.bootContextMap.at(logSlots);

	Ciphertext* tmpvec = new Ciphertext[k];

	NTL_EXEC_RANGE(k, first, last);
	for (long j = first; j < last; ++j) {
		multByPolyNTT(tmpvec[j], rotvec[j], bootContext->rpvec[j], bootContext->bndvec[j], bootContext->logp);
	}
	NTL_EXEC_RANGE_END;

	for (long j = 1; j < k; ++j) {
		addAndEqual(tmpvec[0], tmpvec[j]);
	}

	cipher.copy(tmpvec[0]);
	for (long ki = k; ki < slots; ki += k) {
		NTL_EXEC_RANGE(k, first, last);
		for (long j = first; j < last; ++j) {
			multByPolyNTT(tmpvec[j], rotvec[j], bootContext->rpvec[j + ki], bootContext->bndvec[j + ki], bootContext->logp);
		}
		NTL_EXEC_RANGE_END;
		for (long j = 1; j < k; ++j) {
			addAndEqual(tmpvec[0], tmpvec[j]);
		}
		leftRotateFastAndEqual(tmpvec[0], ki);
		addAndEqual(cipher, tmpvec[0]);
	}
	reScaleByAndEqual(cipher, bootContext->logp);
	delete[] rotvec;
	delete[] tmpvec;
}

void Scheme::slotToCoeffAndEqual(Ciphertext& cipher) {
	long slots = cipher.n;
	long logSlots = log2(slots);
	long logk = logSlots / 2;
	long k = 1 << logk;

	Ciphertext* rotvec = new Ciphertext[k];
	rotvec[0].copy(cipher);

	NTL_EXEC_RANGE(k-1, first, last);
	for (long j = first; j < last; ++j) {
		leftRotateFast(rotvec[j + 1], rotvec[0], j + 1);
	}
	NTL_EXEC_RANGE_END;

	BootContext* bootContext = ring.bootContextMap.at(logSlots);

	Ciphertext* tmpvec = new Ciphertext[k];

	NTL_EXEC_RANGE(k, first, last);
	for (long j = first; j < last; ++j) {
		multByPolyNTT(tmpvec[j], rotvec[j], bootContext->rpvecInv[j], bootContext->bndvecInv[j], bootContext->logp);
	}
	NTL_EXEC_RANGE_END;

	for (long j = 1; j < k; ++j) {
		addAndEqual(tmpvec[0], tmpvec[j]);
	}
	cipher.copy(tmpvec[0]);

	for (long ki = k; ki < slots; ki+=k) {
		NTL_EXEC_RANGE(k, first, last);
		for (long j = first; j < last; ++j) {
			multByPolyNTT(tmpvec[j], rotvec[j], bootContext->rpvecInv[j + ki], bootContext->bndvecInv[j + ki], bootContext->logp);
		}
		NTL_EXEC_RANGE_END;

		for (long j = 1; j < k; ++j) {
			addAndEqual(tmpvec[0], tmpvec[j]);
		}

		leftRotateFastAndEqual(tmpvec[0], ki);
		addAndEqual(cipher, tmpvec[0]);
	}
	reScaleByAndEqual(cipher, bootContext->logp);
	delete[] rotvec;
	delete[] tmpvec;
}

void Scheme::exp2piAndEqual(Ciphertext& cipher, long logp) {
	Ciphertext cipher2;
	square(cipher2, cipher);
	reScaleByAndEqual(cipher2, logp); // cipher2.logq : logq - logp

	Ciphertext cipher4;
	square(cipher4, cipher2);
	reScaleByAndEqual(cipher4, logp); // cipher4.logq : logq -2logp
	RR c = 1/(2*Pi);
	Ciphertext cipher01;
	addConst(cipher01, cipher, c, logp); // cipher01.logq : logq

	c = 2*Pi;
	multByConstAndEqual(cipher01, c, logp);
	reScaleByAndEqual(cipher01, logp); // cipher01.logq : logq - logp

	c = 3/(2*Pi);
	Ciphertext cipher23;
	addConst(cipher23, cipher, c, logp); // cipher23.logq : logq

	c = 4*Pi*Pi*Pi/3;
	multByConstAndEqual(cipher23, c, logp);
	reScaleByAndEqual(cipher23, logp); // cipher23.logq : logq - logp

	multAndEqual(cipher23, cipher2);
	reScaleByAndEqual(cipher23, logp); // cipher23.logq : logq - 2logp

	addAndEqual(cipher23, cipher01); // cipher23.logq : logq - 2logp

	c = 5/(2*Pi);
	Ciphertext cipher45;
	addConst(cipher45, cipher, c, logp); // cipher45.logq : logq

	c = 4*Pi*Pi*Pi*Pi*Pi/15;
	multByConstAndEqual(cipher45, c, logp);
	reScaleByAndEqual(cipher45, logp); // cipher45.logq : logq - logp

	c = 7/(2*Pi);
	addConstAndEqual(cipher, c, logp); // cipher.logq : logq

	c = 8*Pi*Pi*Pi*Pi*Pi*Pi*Pi/315;
	multByConstAndEqual(cipher, c, logp);
	reScaleByAndEqual(cipher, logp); // cipher.logq : logq - logp

	multAndEqual(cipher, cipher2);
	reScaleByAndEqual(cipher, logp); // cipher.logq : logq - 2logp

	modDownByAndEqual(cipher45, logp); // cipher45.logq : logq - 2logp
	addAndEqual(cipher, cipher45); // cipher.logq : logq - 2logp

	multAndEqual(cipher, cipher4);
	reScaleByAndEqual(cipher, logp); // cipher.logq : logq - 3logp

	modDownByAndEqual(cipher23, logp);
	addAndEqual(cipher, cipher23); // cipher.logq : logq - 3logp
}

void Scheme::evalExpAndEqual(Ciphertext& cipher, long logT, long logI) {
	long slots = cipher.n;
	long logSlots = log2(slots);
	BootContext* bootContext = ring.bootContextMap.at(logSlots);
	if(logSlots < logNh) {
		Ciphertext tmp;
		conjugate(tmp, cipher);
		subAndEqual(cipher, tmp);
		divByPo2AndEqual(cipher, logT + 1); // bitDown: logT + 1
		exp2piAndEqual(cipher, bootContext->logp); // bitDown: logT + 1 + 3(logq + logI)
		for (long i = 0; i < logI + logT; ++i) {
			squareAndEqual(cipher);
			reScaleByAndEqual(cipher, bootContext->logp);
		}
		conjugate(tmp, cipher);
		subAndEqual(cipher, tmp);
		multByPolyNTT(tmp, cipher, bootContext->rp1, bootContext->bnd1, bootContext->logp);
		Ciphertext tmprot;
		leftRotateFast(tmprot, tmp, slots);
		addAndEqual(tmp, tmprot);
		multByPolyNTTAndEqual(cipher, bootContext->rp2, bootContext->bnd2, bootContext->logp);
		leftRotateFast(tmprot, cipher, slots);
		addAndEqual(cipher, tmprot);
		addAndEqual(cipher, tmp);
	} else {
		Ciphertext tmp;
		conjugate(tmp, cipher);
		Ciphertext c2;
		sub(c2, cipher, tmp);
		addAndEqual(cipher, tmp);
		imultAndEqual(cipher);
		divByPo2AndEqual(cipher, logT + 1); // cipher bitDown: logT + 1
		reScaleByAndEqual(c2, logT + 1); // c2 bitDown: logT + 1
		exp2piAndEqual(cipher, bootContext->logp); // cipher bitDown: logT + 1 + 3(logq + logI)
		exp2piAndEqual(c2, bootContext->logp); // c2 bitDown: logT + 1 + 3(logq + logI)
		for (long i = 0; i < logI + logT; ++i) {
			squareAndEqual(c2);
			squareAndEqual(cipher);
			reScaleByAndEqual(c2, bootContext->logp);
			reScaleByAndEqual(cipher, bootContext->logp);
		}
		conjugate(tmp, c2);
		subAndEqual(c2, tmp);
		conjugate(tmp, cipher);
		subAndEqual(cipher, tmp);
		imultAndEqual(cipher);
		subAndEqual2(c2, cipher);
		RR c = 0.25/Pi;
		multByConstAndEqual(cipher, c, bootContext->logp);
	}
	reScaleByAndEqual(cipher, bootContext->logp + logI);
}

void Scheme::bootstrapAndEqual(Ciphertext& cipher, long logq, long logQ, long logT, long logI) {
	long logSlots = log2(cipher.n);
	long logp = cipher.logp;

	modDownToAndEqual(cipher, logq);
	normalizeAndEqual(cipher);

	cipher.logq = logQ;
	cipher.logp = logq + 4;
	Ciphertext rot;
	for (long i = logSlots; i < logNh; ++i) {
		leftRotateFast(rot, cipher, (1 << i));
		addAndEqual(cipher, rot);
	}

	divByPo2AndEqual(cipher, logNh); // bitDown: context.logNh - logSlots
	coeffToSlotAndEqual(cipher);
	evalExpAndEqual(cipher, logT, logI); // bitDown: context.logNh + (logI + logT + 5) * logq + (logI + logT + 6) * logI + logT + 1
	slotToCoeffAndEqual(cipher);

	cipher.logp = logp;
}
