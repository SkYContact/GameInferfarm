cmake_minimum_required(VERSION 3.16)
project(inferfarm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
    # /utf-8：源文件中文注释；/W3 默认；多进程编译
    add_compile_options(/utf-8 /MP)
endif()

option(INFERFARM_WITH_TRT "Build the TensorRT backend (needs TRT headers + nvinfer DLL at runtime)" OFF)
set(INFERFARM_TRT_INCLUDE_DIR "D:/ygo_data/trt_lab/include" CACHE PATH
    "Dir containing NvInferRuntime.h (+ minimal cuda_runtime_api.h stub)")

add_library(inferfarm STATIC
    src/fiber_pool.cpp
    src/census.cpp
    src/bank.cpp
    src/refit.cpp
    src/farm.cpp
    src/backends/cpu_backend.cpp
    src/backends/ort_backend.cpp
    src/backends/trt_backend.cpp
)
target_include_directories(inferfarm
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
if (WIN32)
    target_link_libraries(inferfarm PRIVATE winmm)   # timeBeginPeriod(1)
endif()
if (INFERFARM_WITH_TRT)
    target_compile_definitions(inferfarm PRIVATE INFERFARM_WITH_TRT=1)
    target_include_directories(inferfarm PRIVATE ${INFERFARM_TRT_INCLUDE_DIR}
                                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt)
endif()

add_executable(toy examples/toy/toy_main.cpp)
target_link_libraries(toy PRIVATE inferfarm)

add_executable(farm_test tests/farm_test.cpp)
target_link_libraries(farm_test PRIVATE inferfarm)

add_executable(refit_probe tests/refit_probe.cpp)
target_link_libraries(refit_probe PRIVATE inferfarm)

add_executable(gomoku examples/gomoku/gomoku_main.cpp)
target_link_libraries(gomoku PRIVATE inferfarm)

enable_testing()
add_test(NAME farm_test COMMAND farm_test)
