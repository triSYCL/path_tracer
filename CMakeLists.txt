cmake_minimum_required(VERSION 3.16)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++2a -O3 -pthread -fopenmp")

function(declare_RT_test)
    # Setup argument parsing
    set(options USES_OPENCL FORCE_SYNCRONOUS_EXEC XILINX_SPECIFIC)
    set(oneValueArgs TARGET)
    set(multiValueArgs HEADERS SOURCES KERNELS)

    # Parse function arguments
    cmake_parse_arguments(declare_RT_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(Files_SRCS  "${PROJECT_SOURCE_DIR}/${declare_RT_test_TARGET}.cpp")
    set_source_files_properties(${Files_SRCS} PROPERTIES LANGUAGE CXX )
    set (Files_BUILD ${Files_SRCS})
    set(TARGET_NAME "${declare_RT_test_TARGET}")

    find_package(PkgConfig REQUIRED)  
    pkg_check_modules(GTKMM gtkmm-3.0)
    set(SYCL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../include)

    link_directories(
        ${GTKMM_LIBRARY_DIRS}
    )
    include_directories(
        ${GTKMM_INCLUDE_DIRS}
        ${SYCL_INCLUDE_DIR}
    )

    add_executable(${TARGET_NAME} ${Files_BUILD})

    target_link_libraries( ${TARGET_NAME} ${GTKMM_LIBRARIES})
    add_definitions(${GTKMM_CFLAGS})
endfunction(declare_RT_test)

add_subdirectory(RT_SYCL)