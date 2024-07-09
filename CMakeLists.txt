cmake_minimum_required(VERSION 3.12)
project(cuda_python_bindings LANGUAGES CXX CUDA)

# Set policies to handle deprecations
cmake_policy(SET CMP0146 NEW)
cmake_policy(SET CMP0148 NEW)

# Find Python and other necessary packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${Python_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIR})

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 80 86)

# List all your source files here
set(SOURCES
    bindings.cu
    parameters.cu
)

# Add library target
add_library(simulation MODULE ${SOURCES})

# Set properties for shared library
set_target_properties(simulation PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Link against necessary libraries
target_link_libraries(simulation ${Python_LIBRARIES} CUDA::cudart CUDA::cufft pybind11::module)

# Ensure NVCC is used for CUDA files
set_source_files_properties(bindings.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(parameters.cu PROPERTIES LANGUAGE CUDA)

# Add necessary flags (update the architecture flag to a supported one)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /MD -arch=sm_70")

# Ensure proper host compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MDd /GR")

# Add definitions for CUB and other libraries if necessary
add_definitions(-DCUB_STDERR)

# Ensure correct include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
