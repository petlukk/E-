# EaCompiler.cmake — CMake integration for the Eä SIMD compiler
#
# Usage:
#   include(cmake/EaCompiler.cmake)
#   add_ea_kernel(<target> <source.ea>)
#   target_link_libraries(myapp <target>)
#
# This creates an imported static library target from the .ea source.
# The generated .h header is exposed via the target's include directory.

find_program(EA_COMPILER ea REQUIRED)

function(add_ea_kernel target source)
    # Resolve to absolute path
    get_filename_component(source_abs "${source}" ABSOLUTE)
    get_filename_component(stem "${source}" NAME_WE)

    set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${stem}.o")
    set(header_file "${CMAKE_CURRENT_BINARY_DIR}/${stem}.h")

    # Compile .ea → .o + .h
    add_custom_command(
        OUTPUT "${obj_file}" "${header_file}"
        COMMAND "${EA_COMPILER}" "${source_abs}" --header
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS "${source_abs}"
        COMMENT "Compiling Eä kernel: ${source}"
    )

    # Wrap the .o in a static library
    add_library(${target} STATIC "${obj_file}")
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE C)

    # Expose the generated header
    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

    # Ensure the header is generated before anything that depends on this target
    add_custom_target(${target}_header DEPENDS "${header_file}")
    add_dependencies(${target} ${target}_header)
endfunction()
