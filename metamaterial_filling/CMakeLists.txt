cmake_minimum_required(VERSION 3.10)
project(metamaterial_filling)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++14)




# find_package(catkin REQUIRED COMPONENTS
#   roscpp
# )

# find_package(Eigen REQUIRED)

find_package(Threads REQUIRED)

find_package(PCL REQUIRED)
find_package (Threads REQUIRED)

find_package(CGAL REQUIRED COMPONENTS Core)

add_definitions(${PCL_DEFINITIONS})

# catkin_package()

# include_directories(${catkin_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
# include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# include_directories(${PROJECT_SOURCE_DIR}/include/CGAL-5.6.1/include)

# add_executable(filling src/filling.cpp)
# target_link_libraries(filling ${PCL_LIBRARIES})

add_executable(scaleMesh src/scaleMesh.cpp)
target_link_libraries(scaleMesh )

add_executable(rotateMesh src/rotateMesh.cpp)
target_link_libraries(rotateMesh)

add_executable(replaceMesh src/replaceMesh.cpp)
target_link_libraries(replaceMesh)

add_executable(smallModelGeneration src/smallModelGeneration.cpp)
target_link_libraries(smallModelGeneration PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${PCL_LIBRARIES} CGAL::CGAL CGAL::CGAL_Core)

add_executable(innerPointsCalculation src/innerPointsCalculation.cpp)
target_link_libraries(innerPointsCalculation PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${PCL_LIBRARIES} CGAL::CGAL CGAL::CGAL_Core)

add_executable(tetrahedralMeshing src/tetrahedralMeshing.cpp)
target_link_libraries(tetrahedralMeshing  ${PCL_LIBRARIES} CGAL::CGAL CGAL::CGAL_Core)

add_executable(checkIfWatertight src/checkIfWatertight.cpp)
target_link_libraries(checkIfWatertight PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${PCL_LIBRARIES} CGAL::CGAL CGAL::CGAL_Core)

