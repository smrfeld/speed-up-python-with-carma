cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(test)

include_directories(/usr/local/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)

find_library(ARMADILLO_LIB armadillo HINTS /usr/local/lib/ REQUIRED)
find_library(GIBBS_LIB simple_gibbs HINTS /usr/local/lib/ REQUIRED)

add_executable(test src/test.cpp)

target_link_libraries(test PUBLIC ${ARMADILLO_LIB} ${GIBBS_LIB})