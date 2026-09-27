cmake_minimum_required(VERSION 3.16)
project(inferfarm LANGUAGES CXX)

option(INFERFARM_CORO20 "C++20 档：/std:c++20 + 缺省 fiber 后端=fcontext（判决实验，见 docs/design-cpp20-coroutines.md）" OFF)

set(CMAKE_CXX_STANDARD 17)
if (INFERFARM_CORO20)
    set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# 单配置生成器缺省 Release（多配置如 VS 由 --config 决定，不受影响）
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

if (MSVC)
    # /utf-8：源文件中文注释（库+测试+示例）；多进程编译。
    # 生成器表达式限定 CXX/C——ASM_MASM 不认这两个旗标（A4018 噪音）。
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX,C>:/utf-8>" "$<$<COMPILE_LANGUAGE:CXX,C>:/MP>")
endif()

option(INFERFARM_WITH_TRT "Build the TensorRT backend (needs TRT headers + nvinfer DLL at runtime)" OFF)
option(INFERFARM_BUILD_TESTS "Build test executables (farm_test/refit_probe/gomoku_backend_test)" ON)
option(INFERFARM_BUILD_EXAMPLES "Build example executables (toy/gomoku/othello)" ON)

# TRT 头文件目录：仅 INFERFARM_WITH_TRT=ON 时需要。缺省空+find_path 自动发现
# （不把开发机路径写进仓库——换机即静默失效的坑）；自动发现失败=显式报错并
# 指路，绝不静默带上坏路径。
set(INFERFARM_TRT_INCLUDE_DIR "" CACHE PATH
    "Dir containing NvInferRuntime.h (+ minimal cuda_runtime_api.h stub); empty=auto-discover")
mark_as_advanced(INFERFARM_TRT_INCLUDE_DIR)
if (INFERFARM_WITH_TRT)
    if (NOT INFERFARM_TRT_INCLUDE_DIR)
        # env FARM_TRT_INCLUDE_DIR=<含 NvInferRuntime.h 的目录> 可 set 一次
        # 永久生效（机器路径不写进仓）；未设则走系统默认搜索
        set(_trt_hint "")
        if (DEFINED ENV{FARM_TRT_INCLUDE_DIR})
            file(TO_CMAKE_PATH "$ENV{FARM_TRT_INCLUDE_DIR}" _trt_hint)
        endif()
        find_path(INFERFARM_TRT_INCLUDE_DIR NvInferRuntime.h HINTS ${_trt_hint})
    endif()
    if (NOT INFERFARM_TRT_INCLUDE_DIR)
        message(FATAL_ERROR
            "INFERFARM_WITH_TRT=ON 但未找到 NvInferRuntime.h——"
            "-DINFERFARM_TRT_INCLUDE_DIR=<含 NvInferRuntime.h 的目录>，"
            "或 set FARM_TRT_INCLUDE_DIR 后重新 configure")
    endif()
endif()

add_library(inferfarm STATIC
    src/affinity.cpp
    src/fiber_pool.cpp
    src/fiber_backend.cpp
    src/fiber_backend_winfiber.cpp
    src/fiber_backend_fcontext.cpp
    src/census.cpp
    src/bank.cpp
    src/refit.cpp
    src/farm.cpp
    src/backends/cpu_backend.cpp
    src/backends/ort_backend.cpp
    src/backends/trt_backend.cpp
    src/backends/ncnn_backend.cpp
)
if (MSVC)
    # fcontext 后端切换体（MASM64；Windows x64 ABI 面，src/fcontext.asm）
    enable_language(ASM_MASM)
    target_sources(inferfarm PRIVATE src/fcontext.asm)
elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    # fcontext 后端切换体（GNU as；SysV AMD64 ELF 面，src/fcontext_sysv.S
    # ——Linux x64 移植路，判决 24 POSIX 路线）。.S 需预处理（__CET__ 门）
    # ⇒ ASM 编译器钉到 C 编译器（gcc/clang 驱动对 .S 先跑 cpp；裸 as 会把
    # 预处理行当注释吞掉）。其余平台（含 macOS Mach-O）不编译切换体，
    # 工厂返回 nullptr → 后端选择层回退报错（见 fiber_backend.cpp）。
    if(NOT CMAKE_ASM_COMPILER)
        set(CMAKE_ASM_COMPILER "${CMAKE_C_COMPILER}")
    endif()
    enable_language(ASM)
    target_sources(inferfarm PRIVATE src/fcontext_sysv.S)
endif()
if (INFERFARM_CORO20)
    target_compile_definitions(inferfarm PRIVATE INFERFARM_CORO20=1)
endif()
target_include_directories(inferfarm
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
# 公共头不泄露第三方依赖（cache.h 等只用 std）——third_party 恒 PRIVATE，
# 消费者无需任何第三方 include 路径。
if (WIN32)
    target_link_libraries(inferfarm PRIVATE winmm)   # timeBeginPeriod(1)
endif()
target_link_libraries(inferfarm PRIVATE Threads::Threads)
if (NOT WIN32)
    target_link_libraries(inferfarm PRIVATE ${CMAKE_DL_LIBS})   # ort dlopen 面
endif()
if (MSVC)
    target_compile_options(inferfarm PRIVATE /W3)
else()
    target_compile_options(inferfarm PRIVATE -Wall -Wextra)
endif()
if (INFERFARM_WITH_TRT)
    target_compile_definitions(inferfarm PRIVATE INFERFARM_WITH_TRT=1)
    target_include_directories(inferfarm PRIVATE ${INFERFARM_TRT_INCLUDE_DIR}
                                                 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt)
endif()

if (INFERFARM_BUILD_EXAMPLES)
    add_executable(toy examples/toy/toy_main.cpp)
    target_link_libraries(toy PRIVATE inferfarm)

    add_executable(gomoku examples/gomoku/gomoku_main.cpp)
    target_link_libraries(gomoku PRIVATE inferfarm)

    # othello：源码用 Win32 控制台 I/O（io.h/_kbhit），且在途维护中——
    # 构建面先门到 Windows（源码零改动，2026-09-27 Linux 移植批注记）
    if (WIN32)
        add_executable(othello examples/othello/othello_main.cpp)
        target_link_libraries(othello PRIVATE inferfarm)
    endif()
endif()

if (INFERFARM_BUILD_TESTS)
    add_executable(farm_test tests/farm_test.cpp)
    target_link_libraries(farm_test PRIVATE inferfarm)

    add_executable(refit_probe tests/refit_probe.cpp)
    target_link_libraries(refit_probe PRIVATE inferfarm)

    # 真模型可选门：模型工件缺席=SKIP 退 0（无 GPU/CI 不拦路）
    add_executable(gomoku_backend_test tests/gomoku_backend_test.cpp)
    target_link_libraries(gomoku_backend_test PRIVATE inferfarm)

    # 纤程开销拆解微基准（判决实验原语单价账；手动跑，不入 ctest）
    add_executable(fiber_bench tests/fiber_bench.cpp)
    target_link_libraries(fiber_bench PRIVATE inferfarm)


    enable_testing()
    add_test(NAME farm_test COMMAND farm_test)
endif()
