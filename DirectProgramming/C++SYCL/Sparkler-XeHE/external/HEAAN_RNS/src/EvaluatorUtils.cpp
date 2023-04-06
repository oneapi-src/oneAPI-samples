/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#include <cmath>
#include <complex>
#include <cstdlib>

#include "EvaluatorUtils.h"

//----------------------------------------------------------------------------------
//   RANDOM REAL AND COMPLEX NUMBERS
//----------------------------------------------------------------------------------

double EvaluatorUtils::randomReal(double bound)  {
	return (double) rand()/(RAND_MAX) * bound;
}

complex<double> EvaluatorUtils::randomComplex(double bound) {
	complex<double> res;
	res.real(randomReal(bound));
	res.imag(randomReal(bound));
	return res;
}

complex<double> EvaluatorUtils::randomCircle(double anglebound) {
	double angle = randomReal(anglebound);
	complex<double> res;
	res.real(cos(angle * 2 * M_PI));
	res.imag(sin(angle * 2 * M_PI));
	return res;
}

double* EvaluatorUtils::randomRealArray(long size, double bound) {
	double* res = new double[size];
	for (long i = 0; i < size; ++i) {
		res[i] = randomReal(bound);
	}
	return res;
}

complex<double>* EvaluatorUtils::randomComplexArray(long size, double bound) {
	complex<double>* res = new complex<double>[size];
	for (long i = 0; i < size; ++i) {
		res[i] = randomComplex(bound);
	}
	return res;
}

complex<double>* EvaluatorUtils::randomCircleArray(long size, double bound) {
	complex<double>* res = new complex<double>[size];
	for (long i = 0; i < size; ++i) {
		res[i] = randomCircle(bound);
	}
	return res;
}

//----------------------------------------------------------------------------------
//   ROTATIONS
//----------------------------------------------------------------------------------

void EvaluatorUtils::leftRotateAndEqual(complex<double>* vals, const long size, const long rotSize) {
	long remrotSize = rotSize % size;
	if(remrotSize != 0) {
		long divisor = gcd(remrotSize, size);
		long steps = size / divisor;
		for (long i = 0; i < divisor; ++i) {
			complex<double> tmp = vals[i];
			long idx = i;
			for (long j = 0; j < steps - 1; ++j) {
				vals[idx] = vals[(idx + remrotSize) % size];
				idx = (idx + remrotSize) % size;
			}
			vals[idx] = tmp;
		}
	}
}

void EvaluatorUtils::rightRotateAndEqual(complex<double>* vals, const long size, const long rotSize) {
	long remrotSize = rotSize % size;
	long leftremrotSize = (size - remrotSize) % size;
	leftRotateAndEqual(vals, size, leftremrotSize);
}
