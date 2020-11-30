add_executable(sycl-rt main.cpp)
target_include_directories(sycl-rt PRIVATE .)
# sound is a SYCL program
add_sycl_to_target(sycl-rt)
