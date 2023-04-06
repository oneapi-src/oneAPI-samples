/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_TESTSCHEME_H_
#define HEAAN_TESTSCHEME_H_

class TestScheme {
public:


	//----------------------------------------------------------------------------------
	//   STANDARD TESTS
	//----------------------------------------------------------------------------------
	

	static void testEncrypt(long logq, long logp, long logn);
	
	static void testEncryptBySk(long logq, long logp, long logn);
	
	static void testDecryptForShare(long logq, long logp, long logn, long logErrorBound);
	
	static void testEncryptSingle(long logq, long logp);
	
	static void testAdd(long logq, long logp, long logn);
	
	static void testMult(long logq, long logp, long logn);
	
	static void testiMult(long logq, long logp, long logn);


	//----------------------------------------------------------------------------------
	//   ROTATE & CONJUGATE TESTS
	//----------------------------------------------------------------------------------


	static void testRotateFast(long logq, long logp, long logn, long r);

	static void testConjugate(long logq, long logp, long logn);


	//----------------------------------------------------------------------------------
	//   POWER & PRODUCT TESTS
	//----------------------------------------------------------------------------------


	static void testPowerOf2(long logq, long logp, long logn, long logdeg);

	static void testPower(long logq, long logp, long logn, long degree);


	//----------------------------------------------------------------------------------
	//   FUNCTION TESTS
	//----------------------------------------------------------------------------------


	static void testInverse(long logq, long logp, long logn, long steps);

	static void testLogarithm(long logq, long logp, long logn, long degree);

	static void testExponent(long logq, long logp, long logn, long degree);

	static void testExponentLazy(long logq, long logp, long logn, long degree);

	static void testSigmoid(long logq, long logp, long logn, long degree);

	static void testSigmoidLazy(long logq, long logp, long logn, long degree);


	//----------------------------------------------------------------------------------
	//   BOOTSTRAPPING TESTS
	//----------------------------------------------------------------------------------
    

	static void testBootstrap(long logq, long logp, long logn, long logT);

	static void testBootstrapSingleReal(long logq, long logp, long logT);
    
    static void testWriteAndRead(long logq, long logp, long logn);

    
};

#endif
