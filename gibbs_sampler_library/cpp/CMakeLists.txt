cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(simple_gibbs VERSION 0.1.0)

# Include dir
include_directories(/usr/local/include)

# Src
AUX_SOURCE_DIRECTORY(src SRC_FILES)

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include/simple_gibbs_bits")

# Source files
set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/gibbs_sampler.hpp
    ${PROJECT_SOURCE_DIR}/gibbs_sampler.cpp
)

# Set up such that XCode organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

# Required library
find_library(ARMADILLO_LIB armadillo HINTS /usr/local/lib/ REQUIRED)

# Add library
add_library(simple_gibbs SHARED ${SOURCE_FILES})

# Link
target_link_libraries(simple_gibbs PUBLIC ${ARMADILLO_LIB})

# Include directories
target_include_directories(simple_gibbs PRIVATE include/simple_gibbs_bits)

# Install
install(TARGETS simple_gibbs DESTINATION lib)

# Install the headers
install(FILES include/simple_gibbs DESTINATION include)

# Create base directory
install(DIRECTORY include/simple_gibbs_bits DESTINATION include)