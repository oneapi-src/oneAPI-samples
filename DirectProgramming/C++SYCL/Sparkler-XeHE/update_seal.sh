#!/bin/bash
# XeHE repo is installed on the same directory level as xehe_seal_buildtest

cp ./external/SEAL/native/tests/seal/evaluator_gpu.cpp ../xehe_seal_buildtest/native/tests/seal/evaluator_gpu.cpp
cp ./external/SEAL/native/src/seal/util/rlwe.cpp ../xehe_seal_buildtest/native/src/seal/util/
cp ./external/SEAL/native/src/seal/util/ntt.cpp ../xehe_seal_buildtest/native/src/seal/util/
cp ./external/SEAL/native/src/seal/util/polyarithsmallmod.cpp ../xehe_seal_buildtest/native/src/seal/util/
cp ./external/SEAL/native/src/seal/encryptor.cpp ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/ckks.cpp ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/context.cpp ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/decryptor.cpp ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/evaluator.cpp ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/ciphertext.cpp ../xehe_seal_buildtest/native/src/seal/

cp ./external/SEAL/native/src/seal/xehe_seal_plgin.hpp ../xehe_seal_buildtest/native/src/seal/

cp ./external/SEAL/native/src/seal/xehe_seal_plgin.fwd.h ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/evaluator.h ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/plaintext.h ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/ckks.h ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/ciphertext.h ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/dynarray.h ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/decryptor.h ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/context.h ../xehe_seal_buildtest/native/src/seal/
cp ./external/SEAL/native/src/seal/encryptor.h ../xehe_seal_buildtest/native/src/seal/
