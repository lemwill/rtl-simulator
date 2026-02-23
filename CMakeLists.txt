cmake_minimum_required(VERSION 3.20)
project(surge VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Platform ─────────────────────────────────────────────────────────────────
if(APPLE)
    execute_process(
        COMMAND brew --prefix llvm
        OUTPUT_VARIABLE LLVM_BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

# ── LLVM (via llvm-config, avoids cmake target -isystem issues) ──────────────
set(LLVM_CONFIG "${LLVM_BREW_PREFIX}/bin/llvm-config")

execute_process(COMMAND ${LLVM_CONFIG} --version OUTPUT_VARIABLE LLVM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --libdir OUTPUT_VARIABLE LLVM_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --cxxflags OUTPUT_VARIABLE LLVM_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --ldflags OUTPUT_VARIABLE LLVM_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --system-libs OUTPUT_VARIABLE LLVM_SYS_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --libs core support orcjit native nativecodegen OUTPUT_VARIABLE LLVM_LIBS_RAW OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "Found LLVM ${LLVM_VERSION} at ${LLVM_BREW_PREFIX}")

# Extract only -D defines from cxxflags (skip include/opt flags that break homebrew clang)
string(REGEX MATCHALL "-D[^ ]+" LLVM_DEFINES "${LLVM_CXX_FLAGS}")

# ── slang (FetchContent) ────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(slang
    GIT_REPOSITORY https://github.com/MikePopoloski/slang.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
set(SLANG_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(slang)

# Note: CMake 4.x may warn about mimalloc-static having build-tree paths
# in INTERFACE_INCLUDE_DIRECTORIES. This is harmless for FetchContent usage.

# ── Surge executable ────────────────────────────────────────────────────────
add_executable(surge
    src/main.cpp
    src/ir/ir.cpp
    src/ir/builder.cpp
    src/codegen/codegen.cpp
    src/sim/runtime.cpp
    src/trace/vcd_writer.cpp
)

target_include_directories(surge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIR}
)

target_compile_definitions(surge PRIVATE ${LLVM_DEFINES})

# Link slang (cmake target) + LLVM (raw linker flags, no cmake targets)
separate_arguments(LLVM_LD_FLAGS_LIST NATIVE_COMMAND ${LLVM_LD_FLAGS})
separate_arguments(LLVM_LIBS_LIST NATIVE_COMMAND ${LLVM_LIBS_RAW})
separate_arguments(LLVM_SYS_LIBS_LIST NATIVE_COMMAND ${LLVM_SYS_LIBS})

target_link_libraries(surge PRIVATE slang::slang)
target_link_options(surge PRIVATE ${LLVM_LD_FLAGS_LIST})
target_link_libraries(surge PRIVATE ${LLVM_LIBS_LIST} ${LLVM_SYS_LIBS_LIST})

# On macOS with Homebrew clang, ensure we link against Homebrew's libc++
# (system libc++ may be too old and miss symbols like __hash_memory)
if(APPLE AND LLVM_BREW_PREFIX)
    target_link_options(surge PRIVATE -nostdlib++)
    target_link_directories(surge PRIVATE ${LLVM_LIB_DIR}/c++)
    target_link_libraries(surge PRIVATE c++)
endif()

target_compile_options(surge PRIVATE -Wall -Wextra -Wpedantic)
