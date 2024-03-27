# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("armv8a_architecture_optimization_deep_learning")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        native-lib.cpp
        source/FullyConnected.cpp
        source/Pooling.cpp
        source/Convolution.cpp
        source/Testing.cpp
        source/Helpers.cpp
        source/LRN.cpp
        source/Flatten.cpp
        source/Network.cpp
        source/Softmax.cpp
        source/AssetManagement.cpp
        source/Profiler.cpp
        source/LayerThread.cpp)



# add onnxruntime library! (name library -> set lib location -> include header directory
add_library(onnxruntime_training_lib SHARED IMPORTED)
set_target_properties(onnxruntime_training_lib PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/onnxruntime_training/${ANDROID_ABI}/libonnxruntime_training.so)
add_library(onnxruntime_lib SHARED IMPORTED)
set_target_properties(onnxruntime_lib PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/onnxruntime/${ANDROID_ABI}/libonnxruntime.so)



# Linking ARM Compute Library
add_library(arm_compute STATIC IMPORTED)
set_target_properties(arm_compute PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/arm_compute/${ANDROID_ABI}/libarm_compute-static.a)
add_library(arm_compute_core STATIC IMPORTED)
set_target_properties(arm_compute_core PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/arm_compute/${ANDROID_ABI}/libarm_compute_core-static.a)
add_library(arm_compute_graph STATIC IMPORTED)
set_target_properties(arm_compute_graph PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/arm_compute/${ANDROID_ABI}/libarm_compute_graph-static.a)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include/
        ${CMAKE_SOURCE_DIR}/include/arm/
        ${CMAKE_SOURCE_DIR}/include/arm/include
        )

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        android
        log
        onnxruntime_training_lib
        onnxruntime_lib
        arm_compute
        arm_compute_core
        arm_compute_graph)

# Common compiler options
#add_compile_options(-Wall -Wextra -O3 -funroll-loops -ffast-math)
add_compile_options(-Wall -Wextra -O2 -fPIC)

# Compiler-specific options
add_compile_options(-pedantic)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")