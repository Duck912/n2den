# SPDX-License-Identifier: GPL-3.0-or-later

## N2denPGO ##

# ThinLTO and profile-guided optimization with clang (clang-cl on Windows). The approach (clang,
# ThinLTO and an LLVM IR profile trained on gameplay) follows Eglantine's release builds; the
# profile itself is trained on n2den. See docs/PGO.md for the build and training steps.
#
#   N2DEN_THINLTO=ON        compile to LLVM bitcode and optimize across files at link time
#   N2DEN_PGO=GENERATE      instrumented build; writes .profraw files to N2DEN_PGO_PROFRAW_DIR
#                           when the program exits normally
#   N2DEN_PGO=USE           optimize with N2DEN_PGO_PROFILE (an llvm-profdata merge of those files)

option(N2DEN_THINLTO "Enable clang ThinLTO (clang-cl needs lld-link and llvm-lib)" OFF)
set(N2DEN_THINLTO_JOBS "6" CACHE STRING
    "Parallel ThinLTO backend jobs at link time (0 = linker default); each job holds a lot of memory")
set(N2DEN_PGO "OFF" CACHE STRING "Profile-guided optimization: OFF, GENERATE or USE")
set_property(CACHE N2DEN_PGO PROPERTY STRINGS OFF GENERATE USE)
set(N2DEN_PGO_PROFILE "${PROJECT_SOURCE_DIR}/pgo/n2den.profdata" CACHE FILEPATH
    "Merged profile read by N2DEN_PGO=USE")
set(N2DEN_PGO_PROFRAW_DIR "${CMAKE_BINARY_DIR}/profraw" CACHE PATH
    "Where an N2DEN_PGO=GENERATE build writes its .profraw files")

string(TOUPPER "${N2DEN_PGO}" N2DEN_PGO)
if (NOT N2DEN_PGO MATCHES "^(OFF|GENERATE|USE)$")
    message(FATAL_ERROR "N2DEN_PGO must be OFF, GENERATE or USE, not '${N2DEN_PGO}'")
endif()

if (NOT N2DEN_THINLTO AND N2DEN_PGO STREQUAL "OFF")
    return()
endif()

if (NOT CXX_CLANG)
    message(FATAL_ERROR "N2DEN_THINLTO and N2DEN_PGO need clang. On Windows configure with "
                        "clang-cl, lld-link and llvm-lib (see docs/PGO.md).")
endif()
if (ENABLE_LTO)
    message(FATAL_ERROR "Use either ENABLE_LTO or N2DEN_THINLTO, not both")
endif()

# Only C and C++: the resource compiler and assembler must not see these flags.
set(_n2den_lang "$<COMPILE_LANGUAGE:C,CXX>")

if (CXX_CLANG_CL)
    if (NOT CMAKE_LINKER MATCHES "lld-link")
        message(FATAL_ERROR "clang-cl with N2DEN_THINLTO/N2DEN_PGO needs -DCMAKE_LINKER=<path to lld-link.exe> "
                            "(found '${CMAKE_LINKER}')")
    endif()
    if (N2DEN_THINLTO AND NOT CMAKE_AR MATCHES "llvm-lib")
        message(FATAL_ERROR "clang-cl with N2DEN_THINLTO needs -DCMAKE_AR=<path to llvm-lib.exe> to archive "
                            "bitcode (found '${CMAKE_AR}')")
    endif()
endif()

if (N2DEN_THINLTO)
    add_compile_options($<${_n2den_lang}:-flto=thin>)
    set(_n2den_lto_cache "${CMAKE_BINARY_DIR}/thinlto-cache")
    if (CXX_CLANG_CL)
        add_link_options(/lldltocache:${_n2den_lto_cache})
        if (NOT N2DEN_THINLTO_JOBS STREQUAL "0")
            add_link_options(/opt:lldltojobs=${N2DEN_THINLTO_JOBS})
        endif()
    else()
        add_link_options(-flto=thin -fuse-ld=lld -Wl,--thinlto-cache-dir=${_n2den_lto_cache})
        if (NOT N2DEN_THINLTO_JOBS STREQUAL "0")
            add_link_options(-Wl,--thinlto-jobs=${N2DEN_THINLTO_JOBS})
        endif()
    endif()
    message(STATUS "n2den: ThinLTO on (${N2DEN_THINLTO_JOBS} link jobs)")
endif()

if (N2DEN_PGO STREQUAL "GENERATE")
    file(MAKE_DIRECTORY "${N2DEN_PGO_PROFRAW_DIR}")
    # Atomic counters: the CPU, GPU and Vulkan threads run the same code at once.
    add_compile_options($<${_n2den_lang}:-fprofile-generate=${N2DEN_PGO_PROFRAW_DIR}>
                        $<${_n2den_lang}:-fprofile-update=atomic>)
    if (CXX_CLANG_CL)
        # Instrumented objects name clang's profile runtime as a default library; lld-link is run
        # directly, not through the clang driver, so it needs the runtime's folder.
        execute_process(COMMAND "${CMAKE_CXX_COMPILER}" /clang:-print-file-name=clang_rt.profile-x86_64.lib
                        OUTPUT_VARIABLE _n2den_profile_rt OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT EXISTS "${_n2den_profile_rt}")
            message(FATAL_ERROR "clang's profile runtime (clang_rt.profile-x86_64.lib) was not found; "
                                "clang-cl reported '${_n2den_profile_rt}'")
        endif()
        get_filename_component(_n2den_profile_rt_dir "${_n2den_profile_rt}" DIRECTORY)
        add_link_options("/LIBPATH:${_n2den_profile_rt_dir}")
    else()
        add_link_options(-fprofile-generate)
    endif()
    message(STATUS "n2den: instrumented PGO build, profiles go to ${N2DEN_PGO_PROFRAW_DIR}")
elseif (N2DEN_PGO STREQUAL "USE")
    if (NOT EXISTS "${N2DEN_PGO_PROFILE}")
        message(FATAL_ERROR "N2DEN_PGO=USE: no profile at ${N2DEN_PGO_PROFILE} (set N2DEN_PGO_PROFILE)")
    endif()
    # Code changed since training only warns (and is compiled as without a profile).
    add_compile_options($<${_n2den_lang}:-fprofile-use=${N2DEN_PGO_PROFILE}>
                        $<${_n2den_lang}:-Wno-profile-instr-out-of-date>
                        $<${_n2den_lang}:-Wno-profile-instr-unprofiled>
                        $<${_n2den_lang}:-Wno-backend-plugin>)
    # A new profile changes the code of every file: reconfigure when it changes, and put its hash
    # in the flags so that ninja and the compiler cache rebuild instead of reusing old objects.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${N2DEN_PGO_PROFILE}")
    file(MD5 "${N2DEN_PGO_PROFILE}" _n2den_profile_hash)
    add_compile_definitions($<${_n2den_lang}:N2DEN_PGO_PROFILE_HASH=${_n2den_profile_hash}>)
    message(STATUS "n2den: PGO build with ${N2DEN_PGO_PROFILE} (md5 ${_n2den_profile_hash})")
endif()
