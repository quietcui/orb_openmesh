cmake_minimum_required(VERSION 3.16)
project(orb_openmesh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 可以顺便把 /utf-8 也加上，避免中文注释乱码警告（MSVC 专用）
if (MSVC)
    add_compile_options(/utf-8)
endif()

find_package(OpenMesh CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_executable(orb_openmesh
        src/main.cpp
        src/CutMesh.cpp
        src/MeshIO.cpp
        src/Flattening.cpp
        src/MeshUtils.cpp
)

target_link_libraries(orb_openmesh PRIVATE
        OpenMeshCore
        OpenMeshTools
        Eigen3::Eigen
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ?? 关键：为这个 target 定义 _USE_MATH_DEFINES ??
target_compile_definitions(orb_openmesh PRIVATE _USE_MATH_DEFINES)
