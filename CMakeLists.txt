cmake_minimum_required(VERSION 3.25)
project(gidit_classification VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cc")
add_executable(gidit_classification ${SOURCES})
target_link_libraries(gidit_classification PRIVATE wavy_deps)
