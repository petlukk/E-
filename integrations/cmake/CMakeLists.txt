cmake_minimum_required(VERSION 3.16)
project(ea_cmake_example C)

include(cmake/EaCompiler.cmake)

# Compile the Eä kernel
add_ea_kernel(kernel kernel.ea)

# Build the application and link against the kernel
add_executable(app main.c)
target_link_libraries(app kernel m)
