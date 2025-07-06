cmake_minimum_required(VERSION 3.10)

set(project_name yolov11_cpp)
project(${project_name})

set(CMAKE_CXX_STANDARD 17)
set(ONNXRUNTIME_VERSION 1.20.1)
set(CMAKE_OSX_ARCHITECTURES "arm64")

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "/Users/miguel/idwall/iq-lib/onnx-${ONNXRUNTIME_VERSION}")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
    HINTS ${ONNXRUNTIME_DIR}/include
)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_DIR}/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime include directory not found")
endif()
if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime library not found")
endif()

include_directories(
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_LIBS}
)

# add_compile_options(-Wno-dangling-gsl)
add_executable(${project_name}
    main.cpp
    detection_utils.cpp
    image_utils.cpp
)

target_link_libraries(${project_name}
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)
