# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set(libs ${CMAKE_SOURCE_DIR}/..)
include_directories(${libs}/cpp/include)

#------------------------------- import third-party dependencies ------------------------------------#
# Create imported target libcpufeatures
add_library(libcpufeatures STATIC IMPORTED)
set_target_properties(libcpufeatures PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/libcpufeatures.a")

# Create imported target libjpeg-turbo
add_library(libjpeg-turbo STATIC IMPORTED)
set_target_properties(libjpeg-turbo PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/liblibjpeg-turbo.a")

# Create imported target libtiff
add_library(libtiff STATIC IMPORTED)
set_target_properties(libtiff PROPERTIES
        INTERFACE_LINK_LIBRARIES "z"
        )
set_target_properties(libtiff PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/liblibtiff.a")

# Create imported target libwebp
add_library(libwebp STATIC IMPORTED)
set_target_properties(libwebp PROPERTIES
        INTERFACE_LINK_LIBRARIES "libcpufeatures"
        )
set_target_properties(libwebp PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/liblibwebp.a")

# Create imported target libjasper
add_library(libjasper STATIC IMPORTED)
set_target_properties(libjasper PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/liblibjasper.a")

# Create imported target libpng
add_library(libpng STATIC IMPORTED)
set_target_properties(libpng PROPERTIES
        INTERFACE_LINK_LIBRARIES "z"
        )
set_target_properties(libpng PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/liblibpng.a")

# Create imported target IlmImf
add_library(IlmImf STATIC IMPORTED)
set_target_properties(IlmImf PROPERTIES
        INTERFACE_LINK_LIBRARIES "z"
        )
set_target_properties(IlmImf PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/libIlmImf.a")

# Create imported target tbb
add_library(tbb STATIC IMPORTED)
set_target_properties(tbb PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "TBB_USE_GCC_BUILTINS=1;__TBB_GCC_BUILTIN_ATOMICS_PRESENT=1;TBB_SUPPRESS_DEPRECATED_MESSAGES=1"
        INTERFACE_LINK_LIBRARIES "c;m;dl"
        )
set_target_properties(tbb PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/libtbb.a")

# Create imported target libprotobuf
add_library(libprotobuf STATIC IMPORTED)
set_target_properties(libprotobuf PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/liblibprotobuf.a")

# Create imported target quirc
add_library(quirc STATIC IMPORTED)
set_target_properties(quirc PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/libquirc.a")

# Create imported target tegra_hal
add_library(tegra_hal STATIC IMPORTED)
set_target_properties(tegra_hal PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/libtegra_hal.a")

# Create imported target ittnotify
add_library(ittnotify STATIC IMPORTED)
set_target_properties(ittnotify PROPERTIES
        INTERFACE_LINK_LIBRARIES "dl"
        )
set_target_properties(ittnotify PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/libittnotify.a")

#------------------------------- import OpenCV modules ------------------------------------#

# Create imported target opencv_core
add_library(opencv_core STATIC IMPORTED)
set_target_properties(opencv_core PROPERTIES
        INTERFACE_LINK_LIBRARIES "$<LINK_ONLY:dl>;$<LINK_ONLY:m>;$<LINK_ONLY:log>;$<LINK_ONLY:tegra_hal>;$<LINK_ONLY:tbb>;$<LINK_ONLY:z>;$<LINK_ONLY:libcpufeatures>;$<LINK_ONLY:ittnotify>;$<LINK_ONLY:tegra_hal>"
        )
set_target_properties(opencv_core PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/libopencv_core.a")

# Create imported target opencv_flann
add_library(opencv_flann STATIC IMPORTED)
set_target_properties(opencv_flann PROPERTIES
        INTERFACE_LINK_LIBRARIES "$<LINK_ONLY:opencv_core>;opencv_core;$<LINK_ONLY:dl>;$<LINK_ONLY:m>;$<LINK_ONLY:log>;$<LINK_ONLY:tegra_hal>"
        )
set_target_properties(opencv_flann PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/libopencv_flann.a")

# Create imported target opencv_imgproc
add_library(opencv_imgproc STATIC IMPORTED)
set_target_properties(opencv_imgproc PROPERTIES
        INTERFACE_LINK_LIBRARIES "$<LINK_ONLY:opencv_core>;opencv_core;$<LINK_ONLY:dl>;$<LINK_ONLY:m>;$<LINK_ONLY:log>;$<LINK_ONLY:tegra_hal>"
        )
set_target_properties(opencv_imgproc PROPERTIES
        IMPORTED_LOCATION "${libs}/jniLibs/libs/${ANDROID_ABI}/libopencv_imgproc.a")


# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        native-lib
        ${log-lib}

        opencv_core
        opencv_imgproc

        # Links the target library to the log library
        # included in the NDK.
        )