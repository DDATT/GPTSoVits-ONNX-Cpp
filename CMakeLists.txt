# cmake needs this line
cmake_minimum_required(VERSION 3.10)

# Define project name
project(vits)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "v")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

set(OPENJTALK_DIR ${PROJECT_SOURCE_DIR}/openjtalk/openjtalk-native-linux-x86_64)
include_directories(${OPENJTALK_DIR})
add_library(openjtalk-native SHARED IMPORTED)
set_property(TARGET openjtalk-native PROPERTY IMPORTED_LOCATION ${OPENJTALK_DIR}/linux/x86_64/libopenjtalk_native.so)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/main.cpp" "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.c" "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/*.hpp")


add_executable(${PROJECT_NAME} ${SOURCES})


target_include_directories(vits PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} )
target_link_libraries(vits PRIVATE ${ONNX_RUNTIME_LIB} openjtalk-native)
