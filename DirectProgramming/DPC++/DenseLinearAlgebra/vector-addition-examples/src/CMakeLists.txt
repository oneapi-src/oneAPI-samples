# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()
set(SOURCE_FILE vector-addition-examples.cpp)
set(TARGET_NAME vector-addition-examples)

# 
# SECTION 1
# This section defines rules to create a cpu-gpu make target
# This can safely be removed if your project is only targetting FPGAs
#

if(DEFINED CUDA AND(NOT(CUDA EQUAL 0)))
  string(CONCAT COMPILE_FLAGS
    "-fsycl -fsycl-targets=nvptx64-nvidia-cuda"
    " -DSYCL2020_DISABLE_DEPRECATION_WARNINGS"
    " --cuda-path=${NV_HOME}"
    " -Wall"
  )
  set(LINK_FLAGS
    "-fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=${NV_HOME}")
else()
  set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
  set(LINK_FLAGS "-fsycl")
endif()

add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})

# 
# End of SECTION 1
#

