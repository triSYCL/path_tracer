cmake_minimum_required(VERSION 3.16)

project(SYCL-ray-tracer LANGUAGES CXX)

# To run fast...
set(CMAKE_BUILD_TYPE Release)
# ... or to debug
#set(CMAKE_BUILD_TYPE Debug)

# Use triSYCL
include(FindtriSYCL)
# TODO when using graphics later: Use gtkmm-3.0 or later
#find_package(GTK2 3.0 REQUIRED gtkmm)

add_subdirectory(RT_SYCL)
