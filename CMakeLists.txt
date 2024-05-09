cmake_minimum_required(VERSION 3.20)

project(Solid_Sim_Muda LANGUAGES CXX CUDA)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
# set MUDA_BUILD_EXAMPLE to OFF for the subdirectory
set(MUDA_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(muda)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 2.6.x)
FetchContent_MakeAvailable(SFML)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found (include: ${EIGEN3_INCLUDE_DIR}, version: ${EIGEN3_VERSION_STRING})")
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

include_directories(${EIGEN3_INCLUDE_DIRS})

add_subdirectory(simulators)
