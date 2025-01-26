cmake_minimum_required(VERSION 3.8)
project(spqlios)

# read the current version from the manifest file
file(READ "manifest.yaml" manifest)
string(REGEX MATCH "version: +(([0-9]+)\\.([0-9]+)\\.([0-9]+))" SPQLIOS_VERSION_BLAH ${manifest})
#message(STATUS "Version: ${SPQLIOS_VERSION_BLAH}")
set(SPQLIOS_VERSION ${CMAKE_MATCH_1})
set(SPQLIOS_VERSION_MAJOR ${CMAKE_MATCH_2})
set(SPQLIOS_VERSION_MINOR ${CMAKE_MATCH_3})
set(SPQLIOS_VERSION_PATCH ${CMAKE_MATCH_4})
message(STATUS "Compiling spqlios-fft version: ${SPQLIOS_VERSION_MAJOR}.${SPQLIOS_VERSION_MINOR}.${SPQLIOS_VERSION_PATCH}")

#set(ENABLE_SPQLIOS_F128 ON CACHE BOOL "Enable float128 via libquadmath")
set(WARNING_PARANOID ON CACHE BOOL "Treat all warnings as errors")
set(ENABLE_TESTING ON CACHE BOOL "Compiles unittests and integration tests")
set(DEVMODE_INSTALL OFF CACHE BOOL "Install private headers and testlib (mainly for CI)")

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type: Release or Debug" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (WARNING_PARANOID)
	add_compile_options(-Wall -Werror -Wno-unused-command-line-argument)
endif()

message(STATUS "CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(X86 ON)
    set(AARCH64 OFF)
else ()
    set(X86 OFF)
    # set(ENABLE_SPQLIOS_F128 OFF) # float128 are only supported for x86 targets
endif ()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(AARCH64 ON)
endif ()

if (CMAKE_SYSTEM_NAME MATCHES "(Windows)|(MSYS)")
    set(WIN32 ON)
endif ()
if (WIN32)
    #overrides for win32
    set(X86 OFF)
    set(AARCH64 OFF)
    set(X86_WIN32 ON)
else()
    set(X86_WIN32 OFF)
    set(WIN32 OFF)
endif (WIN32)

message(STATUS "--> WIN32: ${WIN32}")
message(STATUS "--> X86_WIN32: ${X86_WIN32}")
message(STATUS "--> X86_LINUX: ${X86}")
message(STATUS "--> AARCH64: ${AARCH64}")

# compiles the main library in spqlios
add_subdirectory(spqlios)

# compiles and activates unittests and itests
if (${ENABLE_TESTING})
    enable_testing()
    add_subdirectory(test)
endif()

