cmake_minimum_required(VERSION 3.16)

option(USE_SINGLE_TASK "Use a SYCL executor that loops over pixel in one task instead of using a parallel_for(), better for FPGA" OFF)
option(SANITIZE_THREADS "Activate thread sanitizer" OFF)
set(SYCL_CXX_COMPILER "" CACHE STRING "Path to the SYCL compiler. Defaults to using triSYCL CPU implementation" )
# Use SYCL host device by default
set(SYCL_DEVICE_TRIPLE "" CACHE STRING "Device triple to be used. only used with SYCL_CXX_COMPILER")
set(SYCL_BACKEND_OPTIONS "" CACHE STRING "Options to pass to the backend compiler")

if (NOT "${SYCL_CXX_COMPILER}" STREQUAL "")
  set(CMAKE_CXX_COMPILER "${SYCL_CXX_COMPILER}" CACHE FILEPATH ${SYCL_CXX_COMPILER} FORCE)
endif()

project(SYCL-path-tracer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use triSYCL
include(FindtriSYCL)

if (NOT "${SYCL_CXX_COMPILER}" STREQUAL "")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Xsycl-target-frontend -fno-exceptions")
  if (NOT "${SYCL_DEVICE_TRIPLE}" STREQUAL "")
    message(STATUS "targeting: ${SYCL_DEVICE_TRIPLE}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=${SYCL_DEVICE_TRIPLE}")
  endif()
  if (NOT "${SYCL_BACKEND_OPTIONS}" STREQUAL "")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend \"${SYCL_BACKEND_OPTIONS}\"")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SYCL_COMPILER")
endif()


# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to Release as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE
    STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(NOT OUTPUT_WIDTH)
  message(STATUS "Setting output width to 800 as none was specified.")
  set(OUTPUT_WIDTH "800" CACHE
	  STRING "Image width in pixel" FORCE)
endif()

if(NOT OUTPUT_HEIGHT)
  message(STATUS "Setting output height to 480 as none was specified.")
  set(OUTPUT_HEIGHT "480" CACHE
	  STRING "Image height in pixel" FORCE)
endif()

if(NOT SAMPLES)
  message(STATUS "Setting samples to 32 as none was specified.")
  set(SAMPLES "32" CACHE
	  STRING "Number of ray that are casted for each pixel" FORCE)
endif()


set(SYCL_RT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SYCL_RT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(sycl-rt ${SYCL_RT_SRC_DIR}/main.cpp)

target_include_directories(sycl-rt PRIVATE ${SYCL_RT_INCLUDE_DIR})
target_compile_definitions(sycl-rt PRIVATE OUTPUT_WIDTH=${OUTPUT_WIDTH})
target_compile_definitions(sycl-rt PRIVATE OUTPUT_HEIGHT=${OUTPUT_HEIGHT})
target_compile_definitions(sycl-rt PRIVATE SAMPLES=${SAMPLES})

# This is a SYCL program
if ("${SYCL_CXX_COMPILER}" STREQUAL "")
  add_sycl_to_target(sycl-rt)
else()
  #target_include_directories(sycl-rt PRIVATE ${TRISYCL_INCLUDE_DIR}/)
endif()

# Use C+20
target_compile_features(sycl-rt PRIVATE cxx_std_20)

if (SANITIZE_THREADS)
target_compile_options(sycl-rt PRIVATE
					   -fno-omit-frame-pointer -fsanitize=thread)
target_link_options(sycl-rt PRIVATE -fsanitize=thread)
endif()
# To use various code sanitizer:
#target_compile_options(sycl-rt PRIVATE
#                       -fno-omit-frame-pointer -fsanitize=address)
#target_link_options(sycl-rt PRIVATE -fsanitize=address)
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

message(STATUS "path_tracer USE_SINGLE_TASK:      ${USE_SINGLE_TASK}")
message(STATUS "path_tracer SANITIZE_THREADS:      ${SANITIZE_THREADS}")
