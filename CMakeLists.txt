cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(uj-proba
        VERSION 0.0.1
	LANGUAGES CXX CUDA)

# Only CPP standard 17 is supported
set(CMAKE_CXX_STANDARD 17)
set(CAFFE2_USE_CUDNN ON)

find_package(Torch REQUIRED)

message(STATUS "LibTorch version: ${Torch_VERSION}")

add_executable(${PROJECT_NAME}
        src/creating_tensors.cc
        src/cuda_basics.cc
        src/math_opers.cc
        src/slicing_reshaping.cc
        src/tensor_vector.cc
        src/main.cc)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBS} ${TORCH_LIBRARIES})
