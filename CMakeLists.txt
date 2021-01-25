cmake_minimum_required(VERSION 3.16)

option(USE_SINGLE_TASK "Use a SYCL executor based on .single_task() instead of .parallel_for(), better for FPGA" OFF)
set(SYCL_CXX_COMPILER "" CACHE STRING "Path to the SYCL compiler. Defaults to using triSYCL CPU implementation" )
# Use SYCL host device by default
set(SYCL_DEVICE_TRIPLE "" CACHE STRING "Device triple to be used. only used with SYCL_CXX_COMPILER")

if (NOT "${SYCL_CXX_COMPILER}" STREQUAL "")
  set(CMAKE_CXX_COMPILER "${SYCL_CXX_COMPILER}")
endif()

project(SYCL-path-tracer LANGUAGES CXX)

# To run fast...
set(CMAKE_BUILD_TYPE Release)
# ... or to debug
#set(CMAKE_BUILD_TYPE Debug)

# Use triSYCL
if ("${SYCL_CXX_COMPILER}" STREQUAL "")
  include(FindtriSYCL)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Xsycl-target-frontend -fno-exceptions")
  if (NOT "${SYCL_DEVICE_TRIPLE}" STREQUAL "")
    message(STATUS "targeting: ${SYCL_DEVICE_TRIPLE}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=${SYCL_DEVICE_TRIPLE}")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SYCL_COMPILER")
endif()

# TODO when using graphics later: Use gtkmm-3.0 or later
#find_package(GTK2 3.0 REQUIRED gtkmm)

add_subdirectory(RT_SYCL)

message(STATUS "path_tracer USE_SINGLE_TASK:      ${USE_SINGLE_TASK}")
