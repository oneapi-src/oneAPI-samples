/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#include "TestScheme.h"
#include "Numb.h"
#include "Context.h"
#include "SecretKey.h"
#include "Scheme.h"
#include "EvaluatorUtils.h"
#include "StringUtils.h"
#include "TimeUtils.h"
#include "SchemeAlgo.h"

#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

void TestScheme::testEncodeSingle(long logN, long L, long logp) {
	cout << "!!! START TEST ENCODE SINGLE !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long k = 1;
	Context context(logN, logp, L, k);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	complex<double> m = EvaluatorUtils::randomCircle();

	timeutils.start("Encrypt single");
	Ciphertext cipher = scheme.encryptSingle(m, L);
	timeutils.stop("Encrypt single");

	timeutils.start("Decrypt single");
	complex<double> d = scheme.decryptSingle(secretKey, cipher);
	timeutils.stop("Decrypt single");

	StringUtils::showcompare(m, d, "val");

	cout << "!!! END TEST ENCODE SINGLE !!!" << endl;
}

void TestScheme::testEncodeBatch(long logN, long L, long logp, long logSlots) {
	cout << "!!! START TEST ENCODE BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long k = L;
	Context context(logN, logp, L, k);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = 1L << logSlots;
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots);

	timeutils.start("Encrypt batch");
	Ciphertext cipher = scheme.encrypt(mvec, slots, L);
	timeutils.stop("Encrypt batch");

	timeutils.start("Decrypt batch");
	complex<double>* dvec = scheme.decrypt(secretKey, cipher);
	timeutils.stop("Decrypt batch");

	StringUtils::showcompare(mvec, dvec, slots, "val");

	cout << "!!! END TEST ENCODE BATCH !!!" << endl;
}

void TestScheme::testBasic(long logN, long L, long logp, long logSlots) {
	cout << "!!! START TEST BASIC !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L + 1;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);

	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = (1 << logSlots);
	double bound = 1.0;
	complex<double>* mvec1 = EvaluatorUtils::randomComplexArray(slots, bound);
	complex<double>* mvec2 = EvaluatorUtils::randomComplexArray(slots, bound);
	complex<double>* cvec = EvaluatorUtils::randomComplexArray(slots, bound);

	complex<double>* mvecAdd = new complex<double>[slots];
	complex<double>* mvecMult = new complex<double>[slots];
	complex<double>* mvecCMult = new complex<double>[slots];

	for(long i = 0; i < slots; i++) {
		mvecAdd[i] = mvec1[i] + mvec2[i];
		mvecMult[i] = mvec1[i] * mvec2[i];
		mvecCMult[i] = mvec1[i] * cvec[i];
	}

	timeutils.start("Encrypt two batch");
	Ciphertext cipher1 = scheme.encrypt(mvec1, slots, L);
	Ciphertext cipher2 = scheme.encrypt(mvec2, slots, L);
	timeutils.stop("Encrypt two batch");

	timeutils.start("Homomorphic Addition");
	Ciphertext addCipher = scheme.add(cipher1, cipher2);
	timeutils.stop("Homomorphic Addition");

	timeutils.start("Homomorphic Multiplication & Rescaling");
	Ciphertext multCipher = scheme.mult(cipher1, cipher2);
	scheme.reScaleByAndEqual(multCipher, 1);
	timeutils.stop("Homomorphic Multiplication & Rescaling");

	timeutils.start("Homomorphic Constant Multiplication & Rescaling");
	Ciphertext cmultCipher = scheme.multByConstVec(cipher1, cvec, slots);
	scheme.reScaleByAndEqual(cmultCipher, 1);
	timeutils.stop("Homomorphic Constant Multiplication & Rescaling");

	timeutils.start("Decrypt batch");
	complex<double>* dvecAdd = scheme.decrypt(secretKey, addCipher);
	complex<double>* dvecCMult = scheme.decrypt(secretKey, cmultCipher);
	complex<double>* dvecMult = scheme.decrypt(secretKey, multCipher);
	timeutils.stop("Decrypt batch");

	StringUtils::showcompare(mvecAdd, dvecAdd, slots, "add");
	StringUtils::showcompare(mvecMult, dvecMult, slots, "mult");
	StringUtils::showcompare(mvecCMult, dvecCMult, slots, "cmult");

}

void TestScheme::testConjugateBatch(long logN, long L, long logp, long logSlots) {
	cout << "!!! START TEST CONJUGATE BATCH !!!" << endl;
	TimeUtils timeutils;
	long K = L;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	//-----------------------------------------
	scheme.addConjKey(secretKey);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = (1 << logSlots);
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots);
	complex<double>* mvecconj = new complex<double>[slots];
	for (long i = 0; i < slots; ++i) {
		mvecconj[i] = conj(mvec[i]);
	}

	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	timeutils.start("Conjugate batch");
	Ciphertext cconj = scheme.conjugate(cipher);
	timeutils.stop("Conjugate batch");

	complex<double>* dvecconj = scheme.decrypt(secretKey, cconj);

	StringUtils::showcompare(mvecconj, dvecconj, slots, "conj");

	cout << "!!! END TEST CONJUGATE BATCH !!!" << endl;
}

void TestScheme::testimultBatch(long logN, long L, long logp, long logSlots) {
	cout << "!!! START TEST i MULTIPLICATION BATCH !!!" << endl;
	TimeUtils timeutils;
	long K = L;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = (1 << logSlots);
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots);
	complex<double>* imvec = new complex<double>[slots];
	for (long i = 0; i < slots; ++i) {
		imvec[i].real(-mvec[i].imag());
		imvec[i].imag(mvec[i].real());
	}

	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	timeutils.start("Multiplication by i batch");
	Ciphertext icipher = scheme.imult(cipher);
	timeutils.stop("Multiplication by i batch");

	complex<double>* idvec = scheme.decrypt(secretKey, icipher);

	StringUtils::showcompare(imvec, idvec, slots, "imult");

	cout << "!!! END TEST i MULTIPLICATION BATCH !!!" << endl;
}

void TestScheme::testRotateByPo2Batch(long logN, long L, long logp, long logRotSlots, long logSlots, bool isLeft) {
	cout << "!!! START TEST ROTATE BY POWER OF 2 BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L + 1;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	scheme.addLeftRotKeys(secretKey);
	scheme.addRightRotKeys(secretKey);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = (1 << logSlots);
	long rotSlots = (1 << logRotSlots);
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots);
	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	if(isLeft) {
		timeutils.start("Left Rotate by power of 2 batch");
		scheme.leftRotateByPo2AndEqual(cipher, logRotSlots);
		timeutils.stop("Left Rotate by power of 2 batch");
	} else {
		timeutils.start("Right Rotate by power of 2 batch");
		scheme.rightRotateByPo2AndEqual(cipher, logRotSlots);
		timeutils.stop("Right Rotate by power of 2 batch");
	}

	complex<double>* dvec = scheme.decrypt(secretKey, cipher);

	if(isLeft) {
		EvaluatorUtils::leftRotateAndEqual(mvec, slots, rotSlots);
	} else {
		EvaluatorUtils::rightRotateAndEqual(mvec, slots, rotSlots);
	}

	StringUtils::showcompare(mvec, dvec, slots, "rot");
	//-----------------------------------------
	cout << "!!! END TEST ROTATE BY POWER OF 2 BATCH !!!" << endl;
}

void TestScheme::testRotateBatch(long logN, long L, long logp, long rotSlots, long logSlots, bool isLeft) {
	cout << "!!! START TEST ROTATE BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	scheme.addLeftRotKeys(secretKey);
	scheme.addRightRotKeys(secretKey);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = (1 << logSlots);
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots);
	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	if(isLeft) {
		timeutils.start("Left rotate batch");
		scheme.leftRotateAndEqual(cipher, rotSlots);
		timeutils.stop("Left rotate batch");
	} else {
		timeutils.start("Right rotate batch");
		scheme.rightRotateAndEqual(cipher, rotSlots);
		timeutils.stop("Right rotate batch");
	}

	complex<double>* dvec = scheme.decrypt(secretKey, cipher);

	if(isLeft) {
		EvaluatorUtils::leftRotateAndEqual(mvec, slots, rotSlots);
	} else {
		EvaluatorUtils::rightRotateAndEqual(mvec, slots, rotSlots);
	}

	StringUtils::showcompare(mvec, dvec, slots, "rot");

	cout << "!!! END TEST ROTATE BATCH !!!" << endl;
}

void TestScheme::testSlotsSum(long logN, long L, long logp, long logSlots) {
	cout << "!!! START TEST SLOTS SUM !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L + 1;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	SchemeAlgo algo(scheme);
	scheme.addLeftRotKeys(secretKey);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = (1 << logSlots);
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots);
	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	timeutils.start("slots sum");
	algo.partialSlotsSumAndEqual(cipher, slots);
	timeutils.stop("slots sum");

	complex<double>* dvec = scheme.decrypt(secretKey, cipher);

	complex<double> msum;
	for (long i = 0; i < slots; ++i) {
		msum += mvec[i];
	}

	StringUtils::showcompare(msum, dvec, slots, "slotsum");

	cout << "!!! END TEST SLOTS SUM !!!" << endl;
}


//----------------------------------------------------------------------------------
//   POWER & PRODUCT TESTS
//----------------------------------------------------------------------------------


void TestScheme::testPowerOf2Batch(long logN, long L, long logp, long logDegree, long logSlots) {
	cout << "!!! START TEST POWER OF 2 BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L + 1;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	SchemeAlgo algo(scheme);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = 1 << logSlots;
	long degree = 1 << logDegree;
	complex<double>* mvec = new complex<double>[slots];
	complex<double>* mpow = new complex<double>[slots];

	for (long i = 0; i < slots; ++i) {
		mvec[i] = EvaluatorUtils::randomCircle();
		mpow[i] = pow(mvec[i], degree);
	}

	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	timeutils.start("Power of 2 batch");
	Ciphertext cpow = algo.powerOf2(cipher, logDegree);
	timeutils.stop("Power of 2 batch");

	complex<double>* dpow = scheme.decrypt(secretKey, cpow);

	StringUtils::showcompare(mpow, dpow, slots, "pow2");

	cout << "!!! END TEST POWER OF 2 BATCH !!!" << endl;
}

//-----------------------------------------

void TestScheme::testPowerBatch(long logN, long L, long logp, long degree, long logSlots) {
	cout << "!!! START TEST POWER BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	SchemeAlgo algo(scheme);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = 1 << logSlots;
	complex<double>* mvec = EvaluatorUtils::randomCircleArray(slots);
	complex<double>* mpow = new complex<double>[slots];
	for (long i = 0; i < slots; ++i) {
		mpow[i] = pow(mvec[i], degree);
	}

	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	timeutils.start("Power batch");
	Ciphertext cpow = algo.power(cipher, degree);
	timeutils.stop("Power batch");

	complex<double>* dpow = scheme.decrypt(secretKey, cpow);

	StringUtils::showcompare(mpow, dpow, slots, "pow");

	cout << "!!! END TEST POWER BATCH !!!" << endl;
}

//-----------------------------------------

void TestScheme::testProdOfPo2Batch(long logN, long L, long logp, long logDegree, long logSlots) {
	cout << "!!! START TEST PROD OF POWER OF 2 BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	SchemeAlgo algo(scheme);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = 1 << logSlots;
	long degree = 1 << logDegree;

	complex<double>** mvec = new complex<double>*[degree];
	for (long i = 0; i < degree; ++i) {
		mvec[i] = EvaluatorUtils::randomCircleArray(slots);
	}

	complex<double>* pvec = new complex<double>[slots];
	for (long j = 0; j < slots; ++j) {
		pvec[j] = mvec[0][j];
		for (long i = 1; i < degree; ++i) {
			pvec[j] *= mvec[i][j];
		}
	}

	Ciphertext* cvec = new Ciphertext[degree];
	for (long i = 0; i < degree; ++i) {
		cvec[i] = scheme.encrypt(mvec[i], slots, L);
	}

	timeutils.start("Product of power of 2 batch");
	Ciphertext cprod = algo.prodOfPo2(cvec, logDegree);
	timeutils.stop("Product of power of 2 batch");

	complex<double>* dvec = scheme.decrypt(secretKey, cprod);

	StringUtils::showcompare(pvec, dvec, slots, "prod");

	cout << "!!! END TEST PROD OF POWER OF 2 BATCH !!!" << endl;
}

void TestScheme::testProdBatch(long logN, long L, long logp, long degree, long logSlots) {
	cout << "!!! START TEST PROD BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	SchemeAlgo algo(scheme);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = 1 << logSlots;
	complex<double>** mvec = new complex<double>*[degree];
	for (long i = 0; i < degree; ++i) {
		mvec[i] = EvaluatorUtils::randomCircleArray(slots);
	}

	complex<double>* pvec = new complex<double>[slots];
	for (long j = 0; j < slots; ++j) {
		pvec[j] = mvec[0][j];
		for (long i = 1; i < degree; ++i) {
			pvec[j] *= mvec[i][j];
		}
	}

	Ciphertext* cvec = new Ciphertext[degree];
	for (long i = 0; i < degree; ++i) {
		cvec[i] = scheme.encrypt(mvec[i], slots, L);
	}

	timeutils.start("Product batch");
	Ciphertext cprod = algo.prod(cvec, degree);
	timeutils.stop("Product batch");

	complex<double>* dvec = scheme.decrypt(secretKey, cprod);

	StringUtils::showcompare(pvec, dvec, slots, "prod");

	cout << "!!! END TEST PROD BATCH !!!" << endl;
}


//----------------------------------------------------------------------------------
//   FUNCTION TESTS
//----------------------------------------------------------------------------------


void TestScheme::testInverseBatch(long logN, long L, long logp, long invSteps, long logSlots) {
	cout << "!!! START TEST INVERSE BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L + 1;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	SchemeAlgo algo(scheme);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = 1 << logSlots;
	complex<double>* mvec = EvaluatorUtils::randomCircleArray(slots, 0.1);
	complex<double>* minv = new complex<double>[slots];
	for (long i = 0; i < slots; ++i) {
		minv[i] = 1. / mvec[i];
	}

	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	timeutils.start("Inverse batch");
	Ciphertext cinv = algo.inverse(cipher, invSteps);
	timeutils.stop("Inverse batch");

	complex<double>* dinv = scheme.decrypt(secretKey, cinv);

	StringUtils::showcompare(minv, dinv, slots, "inv");

	cout << "!!! END TEST INVERSE BATCH !!!" << endl;
}

void TestScheme::testLogarithmBatch(long logN, long L, long logp, long degree, long logSlots) {
	cout << "!!! START TEST LOGARITHM BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	SchemeAlgo algo(scheme);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = 1 << logSlots;
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots, 0.1);
	complex<double>* mlog = new complex<double>[slots];
	for (long i = 0; i < slots; ++i) {
		mlog[i] = log(mvec[i] + 1.);
	}

	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	timeutils.start(LOGARITHM + " batch");
	Ciphertext clog = algo.function(cipher, LOGARITHM, degree);
	timeutils.stop(LOGARITHM + " batch");

	complex<double>* dlog = scheme.decrypt(secretKey, clog);

	StringUtils::showcompare(mlog, dlog, slots, LOGARITHM);

	cout << "!!! END TEST LOGARITHM BATCH !!!" << endl;
}

void TestScheme::testExponentBatch(long logN, long L, long logp, long degree, long logSlots) {
	cout << "!!! START TEST EXPONENT BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	SchemeAlgo algo(scheme);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = 1 << logSlots;
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots);
	complex<double>* mexp = new complex<double>[slots];
	for (long i = 0; i < slots; ++i) {
		mexp[i] = exp(mvec[i]);
	}

	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	timeutils.start(EXPONENT + " batch");
	Ciphertext cexp = algo.exponent(cipher, degree);
	timeutils.stop(EXPONENT + " batch");

	complex<double>* dexp = scheme.decrypt(secretKey, cexp);

	StringUtils::showcompare(mexp, dexp, slots, EXPONENT);

	cout << "!!! END TEST EXPONENT BATCH !!!" << endl;
}

//-----------------------------------------

void TestScheme::testSigmoidBatch(long logN, long L, long logp, long degree, long logSlots) {
	cout << "!!! START TEST SIGMOID BATCH !!!" << endl;
	//-----------------------------------------
	TimeUtils timeutils;
	long K = L + 1;
	Context context(logN, logp, L, K);
	SecretKey secretKey(context);
	Scheme scheme(secretKey, context);
	SchemeAlgo algo(scheme);
	//-----------------------------------------
	srand(time(NULL));
	//-----------------------------------------
	long slots = 1 << logSlots;

	complex<double>* mvec = EvaluatorUtils::randomComplexArray(slots);
	complex<double>* msig = new complex<double>[slots];
	for (long i = 0; i < slots; ++i) {
		msig[i] = exp(mvec[i]) / (1. + exp(mvec[i]));
	}

	Ciphertext cipher = scheme.encrypt(mvec, slots, L);

	timeutils.start(SIGMOID + " batch");
	Ciphertext csig = algo.sigmoid(cipher, degree);
	timeutils.stop(SIGMOID + " batch");

	complex<double>* dsig = scheme.decrypt(secretKey, csig);

	StringUtils::showcompare(msig, dsig, slots, SIGMOID);

	cout << "!!! END TEST SIGMOID BATCH !!!" << endl;
}
