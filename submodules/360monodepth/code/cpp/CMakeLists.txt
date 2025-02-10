CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(InstaOmniDepth)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX AND NOT APPLE)
set(LINUX TRUE)
endif()

# include 3rd libraries path
include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeConfig.txt)
message("OS: ${CMAKE_SYSTEM_NAME}")
message("Generated with config types: ${CMAKE_BUILD_TYPE}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# GTest 
enable_testing()
find_package(GTest REQUIRED)

# EIGEN
find_package(Eigen3 REQUIRED)

# GLog
if (WIN32)
  find_package(glog REQUIRED)
endif()

# boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")

# OpenCV
find_package(OpenCV REQUIRED)

# Ceres
find_package(Ceres REQUIRED)

# Pybind11
set(pybind11_DIR ${PROJECT_SOURCE_DIR}/3rd_party/pybind11/build/mock_install/share/cmake/pybind11 )
find_package(pybind11 REQUIRED)

#=========  depth_stitch.dll =========#
set(MODULE_NAME depth_stitch)

# Source files
file(GLOB sources     ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM sources ${PROJECT_SOURCE_DIR}/src/EigenSolvers.cpp)
file(GLOB headers_hpp ${PROJECT_SOURCE_DIR}/include/*.hpp)
file(GLOB headers_h   ${PROJECT_SOURCE_DIR}/include/*.h)

list(APPEND headers ${headers_h} ${headers_hpp} )
list(APPEND sources ${sources})

add_library(${MODULE_NAME} SHARED
    ${sources}
    ${headers}
)

# add postfix for debug lib
set_target_properties(${MODULE_NAME} PROPERTIES DEBUG_POSTFIX "_d")

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # generate *.pdb file for debug
    message("Set ${MODULE_NAME} to generate *.pdb for MSVC debugging.")
    target_compile_options(${MODULE_NAME} PRIVATE $<$<CONFIG:Release>:/Zi>)
    target_link_options(${MODULE_NAME} PRIVATE $<$<CONFIG:Release>:/DEBUG>)
    target_link_options(${MODULE_NAME} PRIVATE $<$<CONFIG:Release>:/OPT:REF>)
    target_link_options(${MODULE_NAME} PRIVATE $<$<CONFIG:Release>:/OPT:ICF>)
    # DLL export all symbols
    set_target_properties(${MODULE_NAME} PROPERTIES ENABLE_EXPORTS 1)
    set_target_properties(${MODULE_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)
endif()

target_include_directories(${MODULE_NAME} PRIVATE 
  ${PROJECT_SOURCE_DIR}/3rd_party/
  ${PROJECT_SOURCE_DIR}/include/
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS} 
)

target_link_libraries(${MODULE_NAME}
  # glog::glog
  ${GLOG_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  # Ceres::ceres
  ${CERES_LIBRARIES}
)

#========= Python module (Just for Coding) =======#
find_package (Python3  REQUIRED COMPONENTS Interpreter Development NumPy)

message("Python3_FOUND:${Python3_FOUND}")
message("Python3_VERSION:${Python3_VERSION}")
message("Python3_Development_FOUND:${Python3_Development_FOUND}")
message("Python3_LIBRARIES:${Python3_LIBRARIES}")
message("Python3_NumPy_FOUND:${Python3_NumPy_FOUND}")
message("Python3_NumPy_INCLUDE_DIRS:${Python3_NumPy_INCLUDE_DIR}")
message("Python3_NumPy_VERSION:${Python3_NumPy_VERSION}")

set(MODULE_NAME depthmapAlign_pymodule)
add_library(${MODULE_NAME}  SHARED 
    ${PROJECT_SOURCE_DIR}/python/instaOmniDepth/depthmapAlignModule.cpp
)
set_target_properties(${MODULE_NAME}  PROPERTIES SUFFIX ".pyd")

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set_property(TARGET ${MODULE_NAME}  PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:>DLL")
    # target_compile_definitions(depthmapAlignModule PUBLIC Py_LIMITED_API)
endif()

target_include_directories(${MODULE_NAME}  PRIVATE 
  ${Python3_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include/
  ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(${MODULE_NAME} 
  PRIVATE 
  ${Python3_LIBRARIES}
  depth_stitch
  Python3::NumPy
)

#=========  Main =========#
set(MODULE_NAME main)

add_executable(${MODULE_NAME}
  "main_depthmapstitch.cpp"
)

set_target_properties(${MODULE_NAME} PROPERTIES VS_DEBUGGER_ENVIRONMENT "${RUNTIMT_ENV_PATH}")
set_target_properties(${MODULE_NAME} PROPERTIES DEBUG_POSTFIX "_d")

target_include_directories(${MODULE_NAME} PRIVATE 
  ${headers}
  ${PROJECT_SOURCE_DIR}/include/
  ${OpenCV_INCLUDE_DIRS} 
)

target_link_libraries(${MODULE_NAME}
  depth_stitch
  # glog::glog
  ${GLOG_LIBRARIES}
  ${OpenCV_LIBS}
)

#=========  GTest =========#
set(MODULE_NAME unit_test)

file(GLOB test_sources ${PROJECT_SOURCE_DIR}/test/*.cpp)
file(GLOB test_headers ${PROJECT_SOURCE_DIR}/test/*.hpp)

add_executable(${MODULE_NAME}
  ${test_headers}
  ${test_sources}
)

target_include_directories(${MODULE_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
)

target_link_libraries(${MODULE_NAME}
  depth_stitch
  GTest::GTest 
  GTest::Main
  #   glog::glog
  ${GLOG_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  # Ceres::ceres
  ${CERES_LIBRARIES}
)

#=========  Eigen Solvers python binding =========#
pybind11_add_module(EigenSolvers ${PROJECT_SOURCE_DIR}/src/EigenSolvers.cpp )
target_include_directories(EigenSolvers PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(EigenSolvers PRIVATE ${GLOG_LIBRARIES})
#target_link_libraries(EigenSolvers PRIVATE glog::glog)
