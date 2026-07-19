cmake_minimum_required(VERSION 3.22)
project(qnnkit)

set(CMAKE_CXX_STANDARD 17)

set(QNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/qnn)

# Reused Qualcomm QNN SampleApp helpers (tensor setup, dynamic loading, logging,
# platform abstraction). DspLog/ is intentionally excluded (needs HAP_farf.h).
file(GLOB QNN_UTIL_SRC
  ${QNN_DIR}/Utils/*.cpp
  ${QNN_DIR}/WrapperUtils/*.cpp
  ${QNN_DIR}/Log/*.cpp
  ${QNN_DIR}/PAL/src/linux/*.cpp
  ${QNN_DIR}/PAL/src/common/*.cpp)

add_library(qnnkit SHARED
  qnn_jni.cpp
  ${QNN_UTIL_SRC})

target_include_directories(qnnkit PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/QNN
  ${CMAKE_CURRENT_SOURCE_DIR}/include/QNN/HTP
  ${QNN_DIR}
  ${QNN_DIR}/Log
  ${QNN_DIR}/Utils
  ${QNN_DIR}/WrapperUtils
  ${QNN_DIR}/PAL/include)

# The reused SampleApp sources are warning-heavy; don't let them fail the build.
target_compile_options(qnnkit PRIVATE -Wno-everything)

find_library(log-lib log)
target_link_libraries(qnnkit ${log-lib})
