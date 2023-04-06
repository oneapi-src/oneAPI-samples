/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#include "TestScheme.h"

int main() {


//----------------------------------------------------------------------------------
//   STANDARD TESTS
//----------------------------------------------------------------------------------


	TestScheme::testEncrypt(1200, 30, 4);
//	TestScheme::testEncryptSingle(300, 30);
//	TestScheme::testAdd(1200, 30, 4);
//	TestScheme::testMult(1200, 30, 4);
//	TestScheme::testimult(300, 30, 4);


//----------------------------------------------------------------------------------
//   ROTATE & CONJUGATE
//----------------------------------------------------------------------------------


//	TestScheme::testRotateFast(1200, 30, 4, 1);
//	TestScheme::testConjugate(300, 30, 4);


//----------------------------------------------------------------------------------
//   POWER & PRODUCT
//----------------------------------------------------------------------------------


//	TestScheme::testPowerOf2(300, 30, 4, 4);
//	TestScheme::testPower(300, 30, 4, 13);


//----------------------------------------------------------------------------------
//   FUNCTION
//----------------------------------------------------------------------------------


//	TestScheme::testInverse(300, 25, 4, 6);
//	TestScheme::testLogarithm(300, 30, 4, 7);
//	TestScheme::testExponent(300, 30, 4, 7);
//	TestScheme::testSigmoid(300, 30, 4, 7);
//	TestScheme::testSigmoidLazy(300, 30, 4, 7);


//----------------------------------------------------------------------------------
//   OTHER
//----------------------------------------------------------------------------------


//	TestScheme::testCiphertextWriteAndRead(65, 30, 2);
//	TestScheme::testBootstrap(29, 620, 23, 3, 2);
//	TestScheme::testBootstrapSingleReal(29, 620, 23, 2);
//	TestScheme::test();


	return 0;
}
