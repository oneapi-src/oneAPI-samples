/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#ifndef HEAAN_EVALUATORUTILS_H_
#define HEAAN_EVALUATORUTILS_H_

#include "Context.h"

class EvaluatorUtils {
public:

	//----------------------------------------------------------------------------------
	//   RANDOM REAL AND COMPLEX NUMBERS
	//----------------------------------------------------------------------------------

	/**
	 * generate random real value in range (0, bound).
	 */
	static double randomReal(double bound = 1.0);

	/**
	 * generate random complex value with both real and imaginary part in range (0, bound).
	 */
	static complex<double> randomComplex(double bound = 1.0);

	/**
	 * generate random complex value with norm 1 and angle bound in (0, anglebound) in radiant
	 */
	static complex<double> randomCircle(double anglebound = 1.0);

	/**
	 * generate array of random real values in range (0, bound)
	 */
	static double* randomRealArray(long size, double bound = 1.0);

	/**
	 * generate array of random complex values with both real and imaginary part in range (0, bound).
	 */
	static complex<double>* randomComplexArray(long size, double bound = 1.0);

	/**
	 * generate array of random complex values with norm 1 and angle bound in (0, anglebound) in radiant
	 */
	static complex<double>* randomCircleArray(long size, double bound = 1.0);

	//----------------------------------------------------------------------------------
	//   ROTATIONS
	//----------------------------------------------------------------------------------

	/**
	 * left indexes rotation of values
	 * @param[in, out] vals: array of values
	 * @param[in] size: array size
	 * @param[in] rotSize: rotation size
	 */
	static void leftRotateAndEqual(complex<double>* vals, const long size, const long rotSize);

	/**
	 * right indexes rotation of values
	 * @param[in, out] vals: array of values
	 * @param[in] size: array size
	 * @param[in] rotSize: rotation size
	 */
	static void rightRotateAndEqual(complex<double>* vals, const long size, const long rotSize);

};

#endif
