cmake_minimum_required(VERSION 3.20)
project(ode_function)

set(CMAKE_CXX_STANDARD 14)

add_executable(tests src/tests.cpp)
target_include_directories(tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(ode src/main.cpp)
target_include_directories(ode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)