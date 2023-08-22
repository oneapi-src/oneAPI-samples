cmake_minimum_required(VERSION 3.10)

set(TYPES "s" "d" "c" "z")
set(CMAKE_CXX_COMPILER icpx)
set(OPENCL_1_2_1_BSP /opt/a10/inteldevstack/a10_gx_pac_ias_1_2_1_pv/opencl/opencl_bsp)

get_filename_component(HALIDE_TOOLS_PATH          "${RECONFIGURABLE_MATMUL_PATH}/../../tools/Halide" ABSOLUTE)
get_filename_component(CPP_SYCL_FPGA_PATH         "${RECONFIGURABLE_MATMUL_PATH}/../../../.." ABSOLUTE)
get_filename_component(PRODUCTIVE_LIBRARIES_PATH  "${RECONFIGURABLE_MATMUL_PATH}/../.."       ABSOLUTE)
get_filename_component(GOOGLE_TEST_PATH           "${RECONFIGURABLE_MATMUL_PATH}/../../test"  ABSOLUTE)

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    # For whatever reason, it is faster to synthesize with OpenCL 1.2.1 BSP.
    # set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
    set(FPGA_DEVICE "${OPENCL_1_2_1_BSP}:pac_a10")
    set(CLOCK_TARGET "360MHz")
    message(STATUS "FPGA_DEVICE was not specified. "
                   "Configuring the design to run on the default FPGA board ${FPGA_DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). "
                   "Please refer to the README for information on board selection.")
else()
    if((NOT("${FPGA_DEVICE}" STREQUAL "intel_s10sx_pac:pac_s10")) AND (NOT("${FPGA_DEVICE}" STREQUAL "${OPENCL_1_2_1_BSP}:pac_a10")) AND (NOT("${FPGA_DEVICE}" STREQUAL "intel_a10gx_pac:pac_a10")))
        message(FATAL_ERROR "Unsupported device ${FPGA_DEVICE}")
    endif()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")
    set(CLOCK_TARGET "480MHz")
endif()

if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "MKLROOT is not set")
endif()

if(NOT DEFINED ENV{INTELFPGAOCLSDKROOT})
    message(FATAL_ERROR "INTELFPGAOCLSDKROOT is not set")
endif()

# Seed for synthesis
if(NOT DEFINED SEED)
    set(SEED 1)
endif()

###############################################################################
### Create results directories
###############################################################################
file(MAKE_DIRECTORY ${RECONFIGURABLE_MATMUL_PATH}/oneapi)
file(MAKE_DIRECTORY ${RECONFIGURABLE_MATMUL_PATH}/bin)
file(MAKE_DIRECTORY ${RECONFIGURABLE_MATMUL_PATH}/reports)

function(get_halide_type symbol halide_type_name)
    if("${symbol}" STREQUAL "s")
        set(${halide_type_name} "Float(32)" PARENT_SCOPE)
    elseif("${symbol}" STREQUAL "d")
        set(${halide_type_name} "Float(64)" PARENT_SCOPE)
    elseif("${symbol}" STREQUAL "c")
        set(${halide_type_name} "Complex(32)" PARENT_SCOPE)
    else()
        set(${halide_type_name} "Complex(64)" PARENT_SCOPE)
    endif()
endfunction()

function(get_c_type symbol c_type_name)
    if("${symbol}" STREQUAL "s")
        set(${c_type_name} "float" PARENT_SCOPE)
    elseif("${symbol}" STREQUAL "d")
        set(${c_type_name} "double" PARENT_SCOPE)
    elseif("${symbol}" STREQUAL "c")
        set(${c_type_name} "complex32_t" PARENT_SCOPE)
    else()
        set(${c_type_name} "complex64_t" PARENT_SCOPE)
    endif()
endfunction()

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

# For generality, the script here assumes that matrix A, matrix B, matrix C, and the scalars (beta and alpha) can have different types,
# and therefore generates all possible combinations of them. However, not all of the combinations are valid. It is up to the user to
# invoke the right combinations.
set(halide_type_name "unknown type")
foreach(TYPEA ${TYPES})
    get_halide_type(${TYPEA} halide_type_name)
    set(TA ${halide_type_name})
    foreach(TYPEB ${TYPES})
        get_halide_type(${TYPEB} halide_type_name)
        set(TB ${halide_type_name})
        foreach(TYPEC ${TYPES})
            get_halide_type(${TYPEC} halide_type_name)
            set(TC ${halide_type_name})
            string(TOUPPER ${TYPEC} UPPER_TYPEC)
            foreach(TYPE_SCALAR ${TYPES})
                get_c_type(${TYPE_SCALAR} c_type_name)
                set(TS ${c_type_name})
                string(TOUPPER ${TYPE_SCALAR} UPPER_TYPE_SCALAR)
                foreach(SIZE "tiny" "large")
                    foreach(HW "a10" "s10")
                        string(TOUPPER ${SIZE}   UPPER_SIZE)
                        string(TOUPPER ${HW}     UPPER_HW)
            
                        # Generate OneAPI files
                        add_custom_command(OUTPUT  ${RECONFIGURABLE_MATMUL_PATH}/oneapi/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.cpp
                                           COMMAND ${CMAKE_CXX_COMPILER} -DTA=${TA} -DTB=${TB} -DTC=${TC} -DTS=${TS} -DTYPEC_${UPPER_TYPEC} -DTYPE_SCALAR_${UPPER_TYPE_SCALAR} -D${UPPER_SIZE} -D${UPPER_HW}
                                                   -DKERNEL="t2sp::blas::row_major::${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul::${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul"
                                                   -DOUTPUT_FILE="${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.cpp"
                                                   -I${HALIDE_TOOLS_PATH}/include
                                                   -std=gnu++11 -O0 -g
                                                   ${RECONFIGURABLE_MATMUL_PATH}/spec.cpp
                                                   -L${HALIDE_TOOLS_PATH}/lib -lpthread -lz -ldl -lHalide
                                                   -o ${RECONFIGURABLE_MATMUL_PATH}/bin/oneapi_${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}
                                           COMMAND env CLEARCODE=1 ${RECONFIGURABLE_MATMUL_PATH}/bin/oneapi_${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}
                                           COMMAND mv ${RECONFIGURABLE_MATMUL_PATH}/bin/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.cpp ${RECONFIGURABLE_MATMUL_PATH}/oneapi/
                                           DEPENDS ${RECONFIGURABLE_MATMUL_PATH}/spec.cpp
                                                   ${RECONFIGURABLE_MATMUL_PATH}/parameters.h
                                                   ${HALIDE_TOOLS_PATH}/include/Halide.h
                                           WORKING_DIRECTORY ${RECONFIGURABLE_MATMUL_PATH}/bin
                                           COMMENT "Generating OneAPI files"
                                           VERBATIM)
                        add_custom_target(oneapi_${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_MATMUL_PATH}/oneapi/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.cpp
                                                                                                                    compiler_tools)
            
                        # Generate a report
                        file(MAKE_DIRECTORY ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW})
                        add_custom_command(OUTPUT  ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}/report.html
                                                   ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}/resources
                                           COMMAND ${CMAKE_CXX_COMPILER}
                                                   -I${RECONFIGURABLE_MATMUL_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                                   ${RECONFIGURABLE_MATMUL_PATH}/oneapi/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.cpp
                                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                                   -Xshardware -Xstarget=${FPGA_DEVICE}
                                                   -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=${CLOCK_TARGET}
                                                   -fsycl -fintelfpga -fsycl-link=early -o ${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.early
                                           COMMAND cp -rf ${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.early.prj/reports/report.html ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}
                                           COMMAND cp -rf ${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.early.prj/reports/resources   ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}
                                           DEPENDS ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                                   oneapi_${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}
                                           WORKING_DIRECTORY ${RECONFIGURABLE_MATMUL_PATH}/bin
                                           COMMENT "Generating a report"
                                           VERBATIM)
                        add_custom_target(report_${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}/report.html
                                                                                                                    ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}/resources)
            
                        # Generate an image
                        add_custom_command(OUTPUT  ${RECONFIGURABLE_MATMUL_PATH}/bin/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.a
                                                   ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}/acl_quartus_report.txt
                                           COMMAND ${CMAKE_CXX_COMPILER}
                                                   -I${RECONFIGURABLE_MATMUL_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                                   ${RECONFIGURABLE_MATMUL_PATH}/oneapi/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.cpp
                                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                                   -Xshardware -Xstarget=${FPGA_DEVICE}
                                                   -Xsffp-reassociate -Xsffp-contract=fast -Xsdsp-mode=prefer-dsp -Xsprofile -Xsclock=${CLOCK_TARGET} -Xsseed=${SEED}
                                                   -fsycl -fintelfpga -fsycl-link=image -o ${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.a
                                           COMMAND cp -rf ${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.prj/reports/report.html    ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}
                                           COMMAND cp -rf ${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.prj/reports/resources      ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}
                                           COMMAND cp -rf ${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.prj/acl_quartus_report.txt ${RECONFIGURABLE_MATMUL_PATH}/reports/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}
                                           DEPENDS ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                                   ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                                   oneapi_${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}
                                           WORKING_DIRECTORY ${RECONFIGURABLE_MATMUL_PATH}/bin
                                           COMMENT "Generating an image"
                                           VERBATIM)
                        add_custom_target(synthesize_${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW} DEPENDS ${RECONFIGURABLE_MATMUL_PATH}/bin/${TYPEA}${TYPEB}${TYPEC}${TYPE_SCALAR}matmul_${SIZE}_${HW}.a)
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endforeach()

# Generate correctness tests for the current kernel. They assume a tiny systolic array on an FPGA emulator.
add_custom_target(oneapi_targets_for_tests)
foreach(MATMUL ${MATMULS})
    add_dependencies(oneapi_targets_for_tests oneapi_${MATMUL}_tiny_a10)
endforeach()

#set(GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS "")
foreach(MATMUL ${MATMULS})
    set(GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS ${GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS} ${RECONFIGURABLE_MATMUL_PATH}/oneapi/${MATMUL}_tiny_a10.cpp)
endforeach()

add_custom_target(tests)
file(MAKE_DIRECTORY ${CURRENT_KERNEL_PATH}/bin)
file(WRITE ${CURRENT_KERNEL_PATH}/bin/tests.sh
     "#!/bin/bash\nfor i in {0..${MAX_TEST_ID}}; do ${CURRENT_KERNEL_PATH}/bin/test_\$i; done")
execute_process(COMMAND chmod a+x tests.sh
                WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin)

foreach(N RANGE 0 ${MAX_TEST_ID})
    add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/test_${N}
                       COMMAND ${CMAKE_CXX_COMPILER}
                               -I${HALIDE_TOOLS_PATH}/include -I$ENV{MKLROOT}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${CPP_SYCL_FPGA_PATH}/include
                               -I${GOOGLE_TEST_PATH}/include -I${GOOGLE_TEST_PATH}/googletest/include
                               -DTINY -DMKL_ILP64 -DT2SP_NDEBUG -DT2SP_TEST_${N} -DFPGA_EMULATOR
                               -fsycl -fintelfpga -O0 -g
                               -Wno-tautological-constant-compare
                               ${CURRENT_KERNEL_PATH}/test.cpp
                               ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                               ${GENERATED_ONEAPI_SOURCE_FILES_FOR_TESTS}
                               -L${HALIDE_TOOLS_PATH}/lib -L$ENV{MKLROOT}/lib/intel64
                               -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl -lz
                               -L${GOOGLE_TEST_PATH}/googletest/lib
                               -lgtest -lgtest_main
                               -lHalide
                               -o ${CURRENT_KERNEL_PATH}/bin/test_${N}
                       DEPENDS oneapi_targets_for_tests
                               ${CURRENT_KERNEL_PATH}/test.cpp
                               ${CURRENT_KERNEL_PATH}/api.hpp
                               ${RECONFIGURABLE_MATMUL_PATH}/api.hpp
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
    add_custom_target(gen_test_${N} DEPENDS ${CURRENT_KERNEL_PATH}/bin/test_${N})
    add_dependencies(tests gen_test_${N})
endforeach()

# Create targets for demonstrating how to run on real hardware
set(PRECISIONS_BACKUP ${PRECISIONS})
foreach(MATMUL ${MATMULS})
    list(GET PRECISIONS 0 PRECISION)
    list(REMOVE_AT PRECISIONS 0)
    string(TOUPPER ${PRECISION} UPPER_PRECISION)
    foreach(SIZE "tiny" "large")
        string(TOUPPER ${SIZE} UPPER_SIZE)
        foreach(HW "a10" "s10")
            string(TOUPPER ${HW} UPPER_HW)
            add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${HALIDE_TOOLS_PATH}/include -I$ENV{MKLROOT}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${CPP_SYCL_FPGA_PATH}/include
                                       -I${GOOGLE_TEST_PATH}/include -I${GOOGLE_TEST_PATH}/googletest/include
                                       -D${UPPER_SIZE} -D${UPPER_HW}  -DPREFIX_${UPPER_PRECISION}
                                       -DMKL_ILP64 -DT2SP_NDEBUG -DFPGA_HARDWARE
                                       -fsycl -fintelfpga
                                       -c ${CURRENT_KERNEL_PATH}/demo.cpp
                                       -o ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                               DEPENDS ${CURRENT_KERNEL_PATH}/demo.cpp
                                       ${CURRENT_KERNEL_PATH}/api.hpp
                                       ${RECONFIGURABLE_MATMUL_PATH}/api.hpp
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
                                       ${RECONFIGURABLE_MATMUL_PATH}/bin/${MATMUL}_${SIZE}_${HW}.a
                                       -L${HALIDE_TOOLS_PATH}/lib -lHalide -lpthread -lz
                                       -o ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                               DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o
                                       synthesize_${MATMUL}_${SIZE}_${HW}
                               WORKING_DIRECTORY ${CURRENT_KERNEL_PATH}/bin
                               COMMENT "Linking the demo application with the device image"
                               VERBATIM)
            add_custom_target(demo_${PRECISION}${KERNEL}_${SIZE}_${HW} DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW})
        endforeach()
    endforeach()
endforeach()
set(PRECISIONS ${PRECISIONS_BACKUP}) 

# Unsign the synthesized bitstream. Do this if using the opencl 1.2.1 A10
set(PRECISIONS_BACKUP ${PRECISIONS})
foreach(MATMUL ${MATMULS})
    list(GET PRECISIONS 0 PRECISION)
    list(REMOVE_AT PRECISIONS 0)
    string(TOUPPER ${PRECISION} UPPER_PRECISION)
    foreach(SIZE "tiny" "large")
        string(TOUPPER ${SIZE} UPPER_SIZE)
        foreach(HW "a10" "s10")
            string(TOUPPER ${HW} UPPER_HW)
            add_custom_command(OUTPUT  ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.unsigned
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${RECONFIGURABLE_MATMUL_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                       -c ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       -fsycl -fintelfpga
                                       -o halide_runtime_etc.o
                               COMMAND ${CMAKE_CXX_COMPILER}
                                       -I${RECONFIGURABLE_MATMUL_PATH}/oneapi -I${CPP_SYCL_FPGA_PATH}/include -I${PRODUCTIVE_LIBRARIES_PATH}/include -I${HALIDE_TOOLS_PATH}/include
                                       -c ${RECONFIGURABLE_MATMUL_PATH}/oneapi/${MATMUL}_${SIZE}_${HW}.cpp
                                       -fsycl -fintelfpga
                                       -o ${MATMUL}_${SIZE}_${HW}.o
                               COMMAND bash -c "rm -rf *temp_${MATMUL}_${SIZE}_${HW}* signed_${MATMUL}_${SIZE}_${HW}* ${MATMUL}_${SIZE}_${HW}*.aocx ${MATMUL}_${SIZE}_${HW}*.aocx.unsigned"
                               COMMAND $ENV{INTELFPGAOCLSDKROOT}/host/linux64/bin/aocl-extract-aocx
                                       -i ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                                       -o ${RECONFIGURABLE_MATMUL_PATH}/bin/${MATMUL}_${SIZE}_${HW}.aocx
                               # The extract command might generate one or more than one .aocx files. For convenience, use a shell script to hand them.
                               COMMAND bash -c "${PRODUCTIVE_LIBRARIES_PATH}/tools/unsign_bitstreams.sh \
                                                ${MATMUL}_${SIZE}_${HW} \
                                                ${OPENCL_1_2_1_BSP}/linux64/libexec \
                                                ${CMAKE_CXX_COMPILER} \
                                                '${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.o  \
                                                 ${MATMUL}_${SIZE}_${HW}.o \
                                                 halide_runtime_etc.o \
                                                 -L${HALIDE_TOOLS_PATH}/lib -lHalide -lpthread -lz \
                                                 -o ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.unsigned \
                                                 -lOpenCL -lsycl' "
                               DEPENDS ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.cpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/halide_runtime_etc.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/complex_helper.hpp
                                       ${PRODUCTIVE_LIBRARIES_PATH}/include/pipe_wrapper.hpp
                                       demo_${PRECISION}${KERNEL}_${SIZE}_${HW}
                               WORKING_DIRECTORY ${RECONFIGURABLE_MATMUL_PATH}/bin
                               COMMENT "Unsigning the bitstream"
                               VERBATIM)
            add_custom_target(unsign_${PRECISION}${KERNEL}_${SIZE}_${HW} DEPENDS ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}.unsigned)
        endforeach()
    endforeach()
endforeach()
set(PRECISIONS ${PRECISIONS_BACKUP})

# Install pre-generated files
set(PRECISIONS_BACKUP ${PRECISIONS})
foreach(MATMUL ${MATMULS})
    list(GET PRECISIONS 0 PRECISION)
    list(REMOVE_AT PRECISIONS 0)
    foreach(SIZE "tiny" "large")
        foreach(HW "a10" "s10")
            add_custom_target(install_${PRECISION}${KERNEL}_${SIZE}_${HW}
                              COMMAND bash -c "${PRODUCTIVE_LIBRARIES_PATH}/tools/install_pre_gen.sh ${PRECISION}${KERNEL}_${SIZE}_${HW}")
        endforeach()
    endforeach()
endforeach()
set(PRECISIONS ${PRECISIONS_BACKUP})

# Clean generated files
set(PRECISIONS_BACKUP ${PRECISIONS})
foreach(MATMUL ${MATMULS})
    list(GET PRECISIONS 0 PRECISION)
    list(REMOVE_AT PRECISIONS 0)
    foreach(SIZE "tiny" "large")
        foreach(HW "a10" "s10")
            add_custom_target(clean_${PRECISION}${KERNEL}_${SIZE}_${HW}
                              COMMAND bash -c "rm -rf ${CURRENT_KERNEL_PATH}/bin/demo_${PRECISION}${KERNEL}_${SIZE}_${HW}*"
                              COMMAND bash -c "rm -rf ${RECONFIGURABLE_MATMUL_PATH}/bin/*${MATMUL}_${SIZE}_${HW}*"
                              COMMAND bash -c "rm -rf ${RECONFIGURABLE_MATMUL_PATH}/oneapi/${MATMUL}_${SIZE}_${HW}*"
                              COMMAND bash -c "rm -rf ${RECONFIGURABLE_MATMUL_PATH}/reports/${MATMUL}_${SIZE}_${HW}/*")
        endforeach()
    endforeach()
endforeach()
set(PRECISIONS ${PRECISIONS_BACKUP})
