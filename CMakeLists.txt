cmake_minimum_required(VERSION 3.16)

project(SYCL-path-tracer LANGUAGES CXX)

# To run fast...
set(CMAKE_BUILD_TYPE Release)
# ... or to debug
#set(CMAKE_BUILD_TYPE Debug)

option(USE_SINGLE_TASK "Use a SYCL executor based on .single_task() instead of .parallel_for(), better for FPGA" OFF)

# Use triSYCL
include(FindtriSYCL)
# TODO when using graphics later: Use gtkmm-3.0 or later
#find_package(GTK2 3.0 REQUIRED gtkmm)

add_subdirectory(RT_SYCL)

message(STATUS "path_tracer USE_SINGLE_TASK:      ${USE_SINGLE_TASK}")
