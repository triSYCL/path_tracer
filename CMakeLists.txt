cmake_minimum_required(VERSION 3.16)

project(SYCL-ray-tracer LANGUAGES CXX)

# Use C+20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# To use various code sanitizer:
#set (CMAKE_CXX_FLAGS_DEBUG
#     "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
#set (CMAKE_CXX_FLAGS_DEBUG
#     "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=thread")
#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fstack-check -fsanitize=undefined")
#set (CMAKE_CXX_FLAGS_DEBUG
#     "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fstack-check ")


# To run fast...
set(CMAKE_BUILD_TYPE Release)
# ... or to debug
#set(CMAKE_BUILD_TYPE Debug)

# On FPGA use a loop on image pixels instead of a parallel_for
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SINGLE_TASK")

# Use triSYCL
include(FindtriSYCL)
# TODO when using graphics later: Use gtkmm-3.0 or later
#find_package(GTK2 3.0 REQUIRED gtkmm)

add_subdirectory(RT_SYCL)
