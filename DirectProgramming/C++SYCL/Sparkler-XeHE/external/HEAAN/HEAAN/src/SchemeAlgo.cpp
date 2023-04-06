/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#include "SchemeAlgo.h"


void SchemeAlgo::powerOf2(Ciphertext& res, Ciphertext& cipher, long logp, long logDegree) {
	res.copy(cipher);
	for (long i = 0; i < logDegree; ++i) {
		scheme.squareAndEqual(res);
		scheme.reScaleByAndEqual(res, logp);
	}
}

void SchemeAlgo::powerOf2Extended(Ciphertext* res, Ciphertext& cipher, long logp, long logDegree) {
	res[0].copy(cipher);
	for (long i = 1; i < logDegree + 1; ++i) {
		scheme.square(res[i], res[i-1]);
		scheme.reScaleByAndEqual(res[i], logp);
	}
}

//-----------------------------------------

void SchemeAlgo::power(Ciphertext& res, Ciphertext& cipher, long logp, long degree) {
	long logDegree = log2((double)degree);
	long po2Degree = 1 << logDegree;

	powerOf2(res, cipher, logp, logDegree);
	long remDegree = degree - po2Degree;
	Ciphertext tmp;
	if(remDegree > 0) {
		power(tmp, cipher, logp, remDegree);
		long bitsDown = tmp.logq - res.logq;
		scheme.modDownByAndEqual(tmp, bitsDown);
		scheme.multAndEqual(res, tmp);
		scheme.reScaleByAndEqual(res, logp);
	}
}

void SchemeAlgo::powerExtended(Ciphertext* res, Ciphertext& cipher, long logp, long degree) {
	long logDegree = log2((double)degree);
	Ciphertext* cpows = new Ciphertext[logDegree + 1];
	powerOf2Extended(cpows, cipher, logp, logDegree);
	long idx = 0;
	for (long i = 0; i < logDegree; ++i) {
		long powi = (1 << i);
		res[idx++].copy(cpows[i]);
		for (int j = 0; j < powi-1; ++j) {
			long bitsDown = res[j].logq - cpows[i].logq;
			scheme.modDownBy(res[idx], res[j], bitsDown);
			scheme.multAndEqual(res[idx], cpows[i]);
			scheme.reScaleByAndEqual(res[idx++], logp);
		}
	}
	res[idx++].copy(cpows[logDegree]);
	long degree2 = (1 << logDegree);
	for (int i = 0; i < (degree - degree2); ++i) {
		long bitsDown = res[i].logq - cpows[logDegree].logq;
		scheme.modDownBy(res[idx], res[i], bitsDown);
		scheme.multAndEqual(res[idx], cpows[logDegree]);
		scheme.reScaleByAndEqual(res[idx++], logp);
	}
	delete[] cpows;
}

void SchemeAlgo::inverse(Ciphertext& res, Ciphertext& cipher, long logp, long steps) {
	Ciphertext cbar;
	Ciphertext cpow;
	Ciphertext tmp;

	scheme.negate(cbar, cipher);
	scheme.addConstAndEqual(cbar, 1.0, logp);
	cpow.copy(cbar);
	scheme.addConst(tmp, cbar, 1.0, logp);
	scheme.modDownByAndEqual(tmp, logp);
	res.copy(tmp);
	for (long i = 1; i < steps; ++i) {
		scheme.squareAndEqual(cpow);
		scheme.reScaleByAndEqual(cpow, logp);
		tmp.copy(cpow);
		scheme.addConstAndEqual(tmp, 1.0, logp);
		scheme.multAndEqual(tmp, res);
		scheme.reScaleByAndEqual(tmp, logp);
		res.copy(tmp);
	}
}

//-----------------------------------------

void SchemeAlgo::function(Ciphertext& res, Ciphertext& cipher, string& funcName, long logp, long degree) {
	Ciphertext* cpows = new Ciphertext[degree];
	powerExtended(cpows, cipher, logp, degree);

	long dlogp = 2 * logp;

	double* coeffs = taylorCoeffsMap.at(funcName);

	scheme.multByConst(res, cpows[0], coeffs[1], logp);
	scheme.addConstAndEqual(res, coeffs[0], dlogp);

	Ciphertext aixi;
	for (int i = 1; i < degree; ++i) {
		if(abs(coeffs[i + 1]) > 1e-27) {
			scheme.multByConst(aixi, cpows[i], coeffs[i + 1], logp);
			scheme.modDownToAndEqual(res, aixi.logq);
			scheme.addAndEqual(res, aixi);
		}
	}
	scheme.reScaleByAndEqual(res, logp);
}

void SchemeAlgo::functionLazy(Ciphertext& res, Ciphertext& cipher, string& funcName, long logp, long degree) {
	Ciphertext* cpows = new Ciphertext[degree];
	powerExtended(cpows, cipher, logp, degree);

	long dlogp = 2 * logp;

	double* coeffs = taylorCoeffsMap.at(funcName);

	scheme.multByConst(res, cpows[0], coeffs[1], logp);
	scheme.addConstAndEqual(res, coeffs[0], dlogp);

	Ciphertext aixi;
	for (int i = 1; i < degree; ++i) {
		if(abs(coeffs[i + 1]) > 1e-27) {
			scheme.multByConst(aixi, cpows[i], coeffs[i + 1], logp);
			long bitsDown = res.logq - aixi.logq;
			scheme.modDownByAndEqual(res, bitsDown);
			scheme.addAndEqual(res, aixi);
		}
	}
}
