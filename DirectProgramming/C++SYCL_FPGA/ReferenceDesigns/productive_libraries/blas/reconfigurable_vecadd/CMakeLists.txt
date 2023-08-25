cmake_minimum_required(VERSION 3.10)

set(TYPES "s" "d" "c" "z")
set(CMAKE_CXX_COMPILER icpx)
set(OPENCL_1_2_1_BSP /opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp)

get_filename_component(HALIDE_TOOLS_PATH          "${RECONFIGURABLE_VECADD_PATH}/../../tools/Halide" ABSOLUTE)
get_filename_component(CPP_SYCL_FPGA_PATH         "${RECONFIGURABLE_VECADD_PATH}/../../../.." ABSOLUTE)
get_filename_component(PRODUCTIVE_LIBRARIES_PATH  "${RECONFIGURABLE_VECADD_PATH}/../.."       ABSOLUTE)
get_filename_component(GOOGLE_TEST_PATH           "${RECONFIGURABLE_VECADD_PATH}/../../test"  ABSOLUTE)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    # For whatever reason, it is faster to synthesize with OpenCL 1.2.1 BSP.
    # set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
    set(FPGA_DEVICE "${OPENCL_1_2_1_BSP}:pac_a10")
    set(HW "a10")
    set(UPPER_HW "A10")
    message(STATUS "FPGA_DEVICE was not specified. "
                   "Configuring the design to run on the default FPGA board ${FPGA_DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). "
                   "Please refer to the README for information on board selection.")
else()
    if((NOT("${FPGA_DEVICE}" STREQUAL "intel_s10sx_pac:pac_s10")) AND (NOT("${FPGA_DEVICE}" STREQUAL "${OPENCL_1_2_1_BSP}:pac_a10")))
        message(FATAL_ERROR "Unsupported device ${FPGA_DEVICE}")
    endif()
    set(HW "s10")
    set(UPPER_HW "S10")
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")
endif()

if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "MKLROOT is not set")
endif()

if(NOT DEFINED ENV{INTELFPGAOCLSDKROOT})
    message(FATAL_ERROR "INTELFPGAOCLSDKROOT is not set")
endif()

# Seed for synthesis
if(NOT DEFINED SEED)
    set(SEED 7)
else()
    if(NOT ("${SEED}" STREQUAL "7"))
        message(WARNING "Synthesis using a seed other than 7 may cause timing failures (seed = ${SEED})")
    endif()
endif()

###############################################################################
### Create results directories
###############################################################################
file(MAKE_DIRECTORY ${RECONFIGURABLE_VECADD_PATH}/oneapi)
file(MAKE_DIRECTORY ${RECONFIGURABLE_VECADD_PATH}/bin)
file(MAKE_DIRECTORY ${RECONFIGURABLE_VECADD_PATH}/reports)

# Due to restrictions of file size, we have split libHalide.a into parts. Join them into the original binary
add_custom_command(OUTPUT  ${HALIDE_TOOLS_PATH}/lib/libHalide.a
                   COMMAND rm -f libHalide.a
                   COMMAND cat libHalide.part.aa libHalide.part.ab libHalide.part.ac libHalide.part.ad libHalide.part.ae libHalide.part.af libHalide.part.ag >> libHalide.a
                   WORKING_DIRECTORY ${HALIDE_TOOLS_PATH}/lib
                   DEPENDS ${HALIDE_TOOLS_PATH}/lib/libHalide.part.aa
                           ${HALIDE_TOOLS_PATH}/lib/libHalide.part.ab
                           ${HALIDE_TOOLS_PATH}/lib/libHalide.part.ac
                           ${HALIDE_TOOLS_PATH}/lib/libHalide.part.ad
                           ${HALIDE_TOOLS_PATH}/lib/libHalide.part.ae
                           ${HALIDE_TOOLS_PATH}/lib/libHalide.part.af
                           ${HALIDE_TOOLS_PATH}/lib/libHalide.part.ag
                   COMMENT "Getting T2S compiler ready"
                   VERBATIM)
add_custom_target(compiler_tools DEPENDS ${HALIDE_TOOLS_PATH}/lib/libHalide.a)

foreach(TYPEA ${TYPES})
    foreach(SIZE "tiny" "large")
        string(TOUPPER ${SIZE}   UPPER_SIZE)
        string(TOUPPER ${TYPEA}  UPPER_TYPE)
    
        # Generate OneAPI files
        add_custom_command(OUTPUT  ${RECONFIGURABLE_VECADD_PATH}/oneapi/${TYPEA}vecadd_${SIZE}_${HW}.cpp
                           COMMAND ${CMAKE_CXX_COMPILER} 
                                   -DKERNEL="t2sp::blas::row_major::${TYPEA}vecadd::${TYPEA}vecadd"
                                   -DOUTPUT_FILE="${TYPEA}vecadd_${SIZE}_${HW}.cpp"
                                   -I${HALIDE_TOOLS_PATH}/include
                                   -std=gnu++11 -O0 -g
                                   -Wno-tautological-constant-compare
                                   ${RECONFIGURABLE_VECADD_PATH}/spec.cpp
                                   -L${HALIDE_TOOLS_PATH}/lib -lpthread -lz -ldl -lHalide
                                   -D${UPPER_SIZE} -DT2SP_${UPPER_TYPE}VECADD -D${UPPER_HW}
                                   -o ${RECONFIGURABLE_VECADD_PATH}/bin/oneapi_${TYPEA}vecadd_${SIZE}_${HW}
                           COMMAND env CLEARCODE=1 ${RECONFIGURABLE_VECADD_PATH}/bin/oneapi_${TYPEA}vecadd_${SIZE}_${HW}
                           COMMAND mv ${RECONFIGURABLE_VECADD_PATH}/bin/${TYPEA}vecadd_${SIZE}_${HW}.cpp ${RECONFIGURABLE_VECADD_PATH}/oneapi/
                           DEPENDS ${RECONFIGURABLE_VECADD_PATH}/spec.cpp
                                   ${RECONFIGURABLE_VECADD_PATH}/parameters.h
                                   ${HALIDE_TOOLS_PATH}/include/Halide.h
                           WORKING_DIRECTORY ${RECONFIGURABLE_VECADD_PATH}/bin
                           COMMENT "Generating OneAPI files"
                           VERBATIM)
        add_custom_target(oneapi_${TYPEA}vecadd_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_VECADD_PATH}/oneapi/${TYPEA}vecadd_${SIZE}_${HW}.cpp
                                                                       compiler_tools)
    
        # Generate a report
        file(MAKE_DIRECTORY ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW})
        add_custom_command(OUTPUT  ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}/report.html
                                   ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}/resources
                           COMMAND ${CMAKE_CXX_COMPILER}
                                   -I${RECONFIGURABLE_VECADD_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                   ${RECONFIGURABLE_VECADD_PATH}/oneapi/${TYPEA}vecadd_${SIZE}_${HW}.cpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                   -Xshardware -Xstarget=${FPGA_DEVICE}
                                   -Wno-tautological-constant-compare
                                   -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz
                                   -fsycl -fintelfpga -fsycl-link=early -o ${TYPEA}vecadd_${SIZE}_${HW}.early
                           COMMAND cp -rf ${TYPEA}vecadd_${SIZE}_${HW}.early.prj/reports/report.html ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}
                           COMMAND cp -rf ${TYPEA}vecadd_${SIZE}_${HW}.early.prj/reports/resources   ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}
                           DEPENDS ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                   oneapi_${TYPEA}vecadd_${SIZE}_${HW}
                           WORKING_DIRECTORY ${RECONFIGURABLE_VECADD_PATH}/bin
                           COMMENT "Generating a report"
                           VERBATIM)
        add_custom_target(report_${TYPEA}vecadd_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}/report.html
                                                                                              ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}/resources)
    
        # Generate an image
        add_custom_command(OUTPUT  ${RECONFIGURABLE_VECADD_PATH}/bin/${TYPEA}vecadd_${SIZE}_${HW}.a
                                   ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}/acl_quartus_report.txt
                           COMMAND ${CMAKE_CXX_COMPILER}
                                   -I${RECONFIGURABLE_VECADD_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                   ${RECONFIGURABLE_VECADD_PATH}/oneapi/${TYPEA}vecadd_${SIZE}_${HW}.cpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                   -Xshardware -Xstarget=${FPGA_DEVICE}
                                   -Wno-tautological-constant-compare
                                   -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=360MHz -Xsseed=${SEED}
                                   -fsycl -fintelfpga -fsycl-link=image -o ${TYPEA}vecadd_${SIZE}_${HW}.a
                           COMMAND cp -rf ${TYPEA}vecadd_${SIZE}_${HW}.prj/reports/report.html    ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}
                           COMMAND cp -rf ${TYPEA}vecadd_${SIZE}_${HW}.prj/reports/resources      ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}
                           COMMAND cp -rf ${TYPEA}vecadd_${SIZE}_${HW}.prj/acl_quartus_report.txt ${RECONFIGURABLE_VECADD_PATH}/reports/${TYPEA}vecadd_${SIZE}_${HW}
                           DEPENDS ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                   oneapi_${TYPEA}vecadd_${SIZE}_${HW}
                           WORKING_DIRECTORY ${RECONFIGURABLE_VECADD_PATH}/bin
                           COMMENT "Generating an image"
                           VERBATIM)
        add_custom_target(synthesize_${TYPEA}vecadd_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_VECADD_PATH}/bin/${TYPEA}vecadd_${SIZE}_${HW}.a)
    endforeach()
endforeach()

# Generate correctness tests for the current kernel. They assume a tiny systolic array on an FPGA emulator.
add_custom_target(oneapi_targets_for_tests)
foreach(PRECISION ${PRECISIONS})
    add_dependencies(oneapi_targets_for_tests oneapi_${PRECISION}vecadd_tiny_a10)
endforeach()

foreach(PRECISION ${PRECISIONS})
    set(GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS ${GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS} ${RECONFIGURABLE_VECADD_PATH}/oneapi/${PRECISION}vecadd_tiny_a10.cpp)
endforeach()

file(MAKE_DIRECTORY ${CURRENT_KERNEL_PATH}/bin)
file(WRITE ${CURRENT_KERNEL_PATH}/bin/tests.sh "")
execute_process(COMMAND chmod a+x tests.sh
                WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin)
add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/test
                   COMMAND ${CMAKE_CXX_COMPILER}
                           -I${HALIDE_TOOLS_PATH}/include -I$ENV{MKLROOT}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${CPP_SYCL_FPGA_PATH}/include
                           -I${GOOGLE_TEST_PATH}/include -I${GOOGLE_TEST_PATH}/googletest/include
                           -DTINY -DMKL_ILP64 -DT2SP_NDEBUG -DFPGA_EMULATOR
                           -fsycl -fintelfpga
                           -Wno-tautological-constant-compare
                           ${CURRENT_KERNEL_PATH}/test.cpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                           ${GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS}
                           -L${HALIDE_TOOLS_PATH}/lib -L$ENV{MKLROOT}/lib/intel64
                           -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl -lz
                           -L${GOOGLE_TEST_PATH}/googletest/lib
                           -lgtest -lgtest_main
                           -lHalide
                           -o ${CURRENT_KERNEL_PATH}/bin/test
                   COMMAND echo ${CURRENT_KERNEL_PATH}/bin/test > tests.sh && chmod a+x tests.sh
                   DEPENDS oneapi_targets_for_tests
                           ${CURRENT_KERNEL_PATH}/test.cpp
                           ${CURRENT_KERNEL_PATH}/api.hpp
                           ${RECONFIGURABLE_VECADD_PATH}/api.hpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                           ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                           ${HALIDE_TOOLS_PATH}/include/Halide.h
                           ${HALIDE_TOOLS_PATH}/include/HalideRuntime.h
                           ${HALIDE_TOOLS_PATH}/include/HalideBuffer.h
                   WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin
                   COMMENT "Generate tests"
                   VERBATIM)
add_custom_target(tests DEPENDS ${CURRENT_KERNEL_PATH}/bin/test)

# Create targets for demonstrating how to run on real hardware
foreach(PRECISION ${PRECISIONS})
    string(TOUPPER ${PRECISION} UPPER_PRECISION)
    string(TOUPPER ${KERNEL} UPPER_KERNEL)
    foreach(SIZE "tiny" "large")
        string(TOUPPER ${SIZE} UPPER_SIZE)
        add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                           COMMAND ${CMAKE_CXX_COMPILER}
                                   -I${HALIDE_TOOLS_PATH}/include -I$ENV{MKLROOT}/include
                                   -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${CPP_SYCL_FPGA_PATH}/include
                                   -I${GOOGLE_TEST_PATH}/include -I${GOOGLE_TEST_PATH}/googletest/include
                                   -D${UPPER_SIZE} -D${UPPER_HW}
                                   -DMKL_ILP64 -DT2SP_NDEBUG -DT2SP_${UPPER_PRECISION}${UPPER_KERNEL}
                                   -fsycl -fintelfpga
                                   -Wno-tautological-constant-compare
                                   -c ${CURRENT_KERNEL_PATH}/demo.cpp
                                   -o ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                           DEPENDS ${CURRENT_KERNEL_PATH}/demo.cpp
                                   ${CURRENT_KERNEL_PATH}/api.hpp
                                   ${RECONFIGURABLE_VECADD_PATH}/api.hpp
                                   ${HALIDE_TOOLS_PATH}/include/Halide.h
                                   ${HALIDE_TOOLS_PATH}/include/HalideRuntime.h
                                   ${HALIDE_TOOLS_PATH}/include/HalideBuffer.h
                           WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin
                           COMMENT "Generating a demo application on ${HW}"
                           VERBATIM)
        add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                           COMMAND ${CMAKE_CXX_COMPILER}
                                   -fsycl -fintelfpga
                                   ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                                   ${RECONFIGURABLE_VECADD_PATH}/bin/${PRECISION}vecadd_${SIZE}_${HW}.a
                                   -L${HALIDE_TOOLS_PATH}/lib -lHalide -lpthread -lz
                                   -Wno-tautological-constant-compare
                                   -o ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                           DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                   synthesize_${PRECISION}vecadd_${SIZE}_${HW}
                           WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin
                           COMMENT "Linking the demo application with the device image"
                           VERBATIM)
        add_custom_target(demo_${PRECISION}${KERNEL}_${SIZE}_${HW} DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW})
    endforeach()
endforeach()

# Unsign the synthesized bitstream. Do this if using the opencl 1.2.1 A10
foreach(PRECISION ${PRECISIONS})
    foreach(SIZE "tiny" "large")
        string(TOUPPER ${SIZE} UPPER_SIZE)
        add_custom_command(OUTPUT  ${RECONFIGURABLE_VECADD_PATH}/bin/${PRECISION}vecadd_${SIZE}_${HW}.unsigned.aocx
                           COMMAND $ENV{INTELFPGAOCLSDKROOT}/host/linux64/bin/aocl-extract-aocx
                                   -i ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                                   -o ${RECONFIGURABLE_VECADD_PATH}/bin/${PRECISION}vecadd_${SIZE}_${HW}.aocx
                           COMMAND bash -c "rm -rf temp* signed*" # clean up
                           COMMAND bash -c "yes Y | source ${OPENCL_1_2_1_BSP}/linux64/libexec/sign_aocx.sh -H openssl_manager \
                                   -i ${PRECISION}vecadd_${SIZE}_${HW}.aocx -r NULL -k NULL \
                                   -o ${PRECISION}vecadd_${SIZE}_${HW}.unsigned.aocx"
                           DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                           WORKING_DIRECTORY ${RECONFIGURABLE_VECADD_PATH}/bin
                           COMMENT "Extracting and unsigning the bitstream"
                           VERBATIM)
        add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.unsigned
                           COMMAND ${CMAKE_CXX_COMPILER}
                                   -I${RECONFIGURABLE_VECADD_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include
                                   -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                   -c ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                   -Wno-tautological-constant-compare
                                   -fsycl -fintelfpga -o halide_runtime_etc.o
                           COMMAND ${CMAKE_CXX_COMPILER}
                                   -I${RECONFIGURABLE_VECADD_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include
                                   -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                   -c ${RECONFIGURABLE_VECADD_PATH}/oneapi/${PRECISION}vecadd_${SIZE}_${HW}.cpp
                                   -Wno-tautological-constant-compare
                                   -fsycl -fintelfpga -o ${PRECISION}vecadd_${SIZE}_${HW}.o
                           COMMAND ${CMAKE_CXX_COMPILER}
                                   # sign_aocx.sh seems to have a bug: its input seems to be turned into unsigned,
                                   # while its output seems signed, as can be verified by running aocl program
                                   -fsycl -fsycl-add-targets=spir64_fpga-unknown-unknown:${PRECISION}vecadd_${SIZE}_${HW}.aocx
                                   ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                                   ${PRECISION}vecadd_${SIZE}_${HW}.o
                                   halide_runtime_etc.o
                                   -L${HALIDE_TOOLS_PATH}/lib -lHalide -lpthread -lz
                                   -o ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.unsigned
                                   -lOpenCL -lsycl
                           DEPENDS ${RECONFIGURABLE_VECADD_PATH}/bin/${PRECISION}vecadd_${SIZE}_${HW}.unsigned.aocx
                                   ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                   demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                           WORKING_DIRECTORY ${RECONFIGURABLE_VECADD_PATH}/bin
                           COMMENT "Adding the bitstream back to the image"
                           VERBATIM)
        add_custom_target(unsign_${PRECISION}${KERNEL}_${SIZE}_${HW} DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.unsigned)
    endforeach()
endforeach()

# Install pre-generated files
foreach(PRECISION ${PRECISIONS})
    foreach(SIZE "tiny" "large")
        add_custom_target(install_${PRECISION}${KERNEL}_${SIZE}_${HW}
                          COMMAND bash -c "${PRODUCTIVE_LIBRARIES_PATH}/tools/install_pre_gen.sh ${PRECISION}${KERNEL}_${SIZE}_${HW}")
    endforeach()
endforeach()

# Clean generated files
foreach(PRECISION ${PRECISIONS})
    foreach(SIZE "tiny" "large")
        add_custom_target(clean_${PRECISION}${KERNEL}_${SIZE}_${HW}
                          COMMAND bash -c "rm -rf ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}*"
                          COMMAND bash -c "rm -rf ${RECONFIGURABLE_VECADD_PATH}/bin/*${PRECISION}vecadd_${SIZE}_${HW}* "
                          COMMAND bash -c "rm -rf ${RECONFIGURABLE_VECADD_PATH}/oneapi/${PRECISION}vecadd_${SIZE}_${HW}*"
                          COMMAND bash -c "rm -rf ${RECONFIGURABLE_VECADD_PATH}/reports/${PRECISION}vecadd_${SIZE}_${HW}/*")
    endforeach()
endforeach()
