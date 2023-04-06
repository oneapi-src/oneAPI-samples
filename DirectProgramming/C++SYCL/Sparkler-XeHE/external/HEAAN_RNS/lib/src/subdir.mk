################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Ciphertext.cpp \
../src/Context.cpp \
../src/EvaluatorUtils.cpp \
../src/Key.cpp \
../src/Numb.cpp \
../src/Plaintext.cpp \
../src/Scheme.cpp \
../src/SchemeAlgo.cpp \
../src/SecretKey.cpp \
../src/StringUtils.cpp \
../src/TestScheme.cpp \
../src/TimeUtils.cpp 

OBJS += \
./src/Ciphertext.o \
./src/Context.o \
./src/EvaluatorUtils.o \
./src/Key.o \
./src/Numb.o \
./src/Plaintext.o \
./src/Scheme.o \
./src/SchemeAlgo.o \
./src/SecretKey.o \
./src/StringUtils.o \
./src/TestScheme.o \
./src/TimeUtils.o 

CPP_DEPS += \
./src/Ciphertext.d \
./src/Context.d \
./src/EvaluatorUtils.d \
./src/Key.d \
./src/Numb.d \
./src/Plaintext.d \
./src/Scheme.d \
./src/SchemeAlgo.d \
./src/SecretKey.d \
./src/StringUtils.d \
./src/TestScheme.d \
./src/TimeUtils.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include -O3 -c -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


