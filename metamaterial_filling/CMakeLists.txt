cmake_minimum_required(VERSION 3.10)
project(metamaterial_filling)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++14)

find_package(Threads REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(scaleMesh src/scaleMesh.cpp)
target_link_libraries(scaleMesh )

add_executable(rotateMesh src/rotateMesh.cpp)
target_link_libraries(rotateMesh)

add_executable(replaceMesh src/replaceMesh.cpp)
target_link_libraries(replaceMesh)

find_package(PCL QUIET)
find_package(CGAL QUIET COMPONENTS Core)

if(PCL_FOUND)
  add_definitions(${PCL_DEFINITIONS})
  include_directories(${PCL_INCLUDE_DIRS})
endif()

if(PCL_FOUND AND CGAL_FOUND)
  add_executable(smallModelGeneration src/smallModelGeneration.cpp)
  target_link_libraries(smallModelGeneration PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${PCL_LIBRARIES} CGAL::CGAL CGAL::CGAL_Core)

  add_executable(innerPointsCalculation src/innerPointsCalculation.cpp)
  target_link_libraries(innerPointsCalculation PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${PCL_LIBRARIES} CGAL::CGAL CGAL::CGAL_Core)

  add_executable(tetrahedralMeshing src/tetrahedralMeshing.cpp)
  target_link_libraries(tetrahedralMeshing  ${PCL_LIBRARIES} CGAL::CGAL CGAL::CGAL_Core)

  add_executable(checkIfWatertight src/checkIfWatertight.cpp)
  target_link_libraries(checkIfWatertight PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${PCL_LIBRARIES} CGAL::CGAL CGAL::CGAL_Core)
else()
  message(WARNING "PCL/CGAL not found. Building only scaleMesh, rotateMesh, and replaceMesh.")
endif()
