cmake_minimum_required(VERSION 3.10)
project(Sitching)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDA 11 REQUIRED)

# pugixml
find_package(pugixml REQUIRED)

# OpenCV
find_package(OpenCV 4.8 REQUIRED)

# TBB
find_package(TBB REQUIRED)

#Yaml
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(    
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    # ${TBB_INCLUDE_DIRS}
)

# Header files
set(PROJECT_HEADERS
    include/inference.h
    include/imagecropper.h
    include/findfeature.h
    include/pairwisematches.h
    include/filterimage.h
    include/estimator.h
    include/adjusterimage.h
    include/focalsdecide.h
    include/warpingimage.h
    include/exposcompensation.h
    include/findseam.h
    include/imagescaler.h
    include/imageblender.h
    include/runyolo.h
    include/stitchingloop.h
)

# Source files
set(PROJECT_SOURCES
    src/imagecropper.cpp
    src/inference.cpp
    src/findfeature.cpp
    src/pairwisematches.cpp
    src/filterimage.cpp
    src/estimator.cpp
    src/adjusterimage.cpp
    src/focalsdecide.cpp
    src/warpingimage.cpp
    src/exposcompensation.cpp
    src/findseam.cpp
    src/imagescaler.cpp
    src/imageblender.cpp
    src/runyolo.cpp
    src/stitchingloop.cpp    
)

# Create the executable
# add_executable(Stitching ${PROJECT_SOURCES} ${PROJECT_HEADERS})
# target_link_libraries(Stitching ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES} pugixml TBB::tbb)

add_executable(Stitching src/Stitching_dev_yolov8_funtion.cpp ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_link_libraries(Stitching ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES} pugixml TBB::tbb)

add_executable(Org_all src/Stitching_dev_yolov8_funtion_all.cpp ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_link_libraries(Org_all ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES} pugixml TBB::tbb)