# Notification of Use of HEaaN Library

As usual HE schemes, [Cheon-Kim-Kim-Song (CKKS)](https://eprint.iacr.org/2016/421.pdf) scheme is secure under the situation that the decryption value is not revealed to the other party. However, the sharing decryption value may cause the leakage of the secret key when using approximate HE schemes including the CKKS scheme. So you have to distinguish between those cases, one to use the decryption algorithm and do not share the value, and the other to modify the decryption algorithm.


## Decryption For Sharing

Our modified decryption algorithm is implemented in the function `Scheme::decryptForShare`. The function `Scheme::decryptForShare` takes additional input `logErrBound`, which is the log value of estimated error bound. This error bound is computed by a party who evaluates a circuit with ciphertexts.  While processing the evaluation, the party simultaneously computes error bound for the same operation, and she shares the computed error bound to the secret key owner for decryption. Then the secret key onwer decrypts the ciphertext using the error bound by the algorithm `Scheme::decryptForShare`. The higher error bound would ensure the stronger security but also occurs the more precision loss.

We warn that our default error bound in the code in `Scheme::decryptForShare` is for the ciphertext satisfying the following conditions:
1. encrypted by private key encryption algorithm (which is implemented by function `Scheme::encryptBySk`)
1. contains fresh encryption error, i.e. no homomorphic operation is evaluated

## How to Test `decryptForShare`

We add the test function for our new decryption algorithm in run/test.cpp. Follow the same method as described in README, but it needs the additional input, as
```
    ./testHEAAN DecryptForShare logB
```
`logB` is the log of error bound B. If you want to use the default value of B, then let `logB` by -1.

