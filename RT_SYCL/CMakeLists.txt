# Produce sycl-rt executable from the sources
add_executable(sycl-rt main.cpp)

target_include_directories(sycl-rt PRIVATE .)

# This is a SYCL program
add_sycl_to_target(sycl-rt)

# Use C+20
target_compile_features(sycl-rt PRIVATE cxx_std_20)

# To use various code sanitizer:
#target_compile_options(sycl-rt PRIVATE
#                       -fno-omit-frame-pointer -fsanitize=address)
#target_link_options(sycl-rt PRIVATE -fsanitize=address)
#target_compile_options(sycl-rt PRIVATE
#                       -fno-omit-frame-pointer -fsanitize=thread)
#target_link_options(sycl-rt PRIVATE -fsanitize=thread)
#target_compile_options(sycl-rt PRIVATE
#                       -fno-omit-frame-pointer -fsanitize=undefined)
#target_link_options(sycl-rt PRIVATE -fsanitize=undefined)
#target_compile_options(sycl-rt PRIVATE
#                       -fno-omit-frame-pointer -fstack-check)
#target_link_options(sycl-rt PRIVATE -fstack-check)


if(USE_SINGLE_TASK)
  # On FPGA use a loop on image pixels instead of a parallel_for
  set_property(TARGET sycl-rt
    APPEND PROPERTY
    COMPILE_DEFINITIONS USE_SINGLE_TASK=)
endif()
