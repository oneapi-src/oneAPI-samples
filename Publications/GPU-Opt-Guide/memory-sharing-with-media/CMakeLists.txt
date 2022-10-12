find_package(PkgConfig)
pkg_check_modules(VA libva libva-drm)
pkg_check_modules(LEVEL_ZERO libze_loader)

if (VA_FOUND AND LEVEL_ZERO_FOUND)

set(name memory-sharing-vaapi)
add_executable(${name} ${name}.cpp)
target_compile_options(${name} PRIVATE ${COMMON_CXX_FLAGS})
target_link_libraries(${name}
PRIVATE
        ${VA_LIBRARIES}
        ${LEVEL_ZERO_LIBRARIES}
)

endif()
