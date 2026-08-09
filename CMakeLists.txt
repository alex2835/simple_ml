cmake_minimum_required(VERSION 3.25)
project(digit_classification VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MNIST PATHS extern/mnist REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cc")
add_executable(digit_classification ${SOURCES})
target_link_libraries(digit_classification PRIVATE wavy_deps)
target_include_directories(digit_classification PRIVATE ${MNIST_INCLUDE_DIR})
target_compile_definitions(digit_classification PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
