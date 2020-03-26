cmake_minimum_required(VERSION 3.15)
project(GPRL_CPP)

set(CMAKE_CXX_STANDARD 14)
find_package(pybind11 REQUIRED)

add_executable(GPRL_CPP main.cpp Tree_node.h Individual.h Tree_node.cpp Individual.cpp Parameter.h Random.h Random.cpp RL_OP.cpp RL_OP.h)
target_link_libraries(GPRL_CPP PRIVATE pybind11::embed)
