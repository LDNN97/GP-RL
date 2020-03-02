cmake_minimum_required(VERSION 3.15)
project(GP_CPP)

set(CMAKE_CXX_STANDARD 14)

add_executable(GP_CPP main.cpp Tree_node.h Individual.h Tree_node.cpp Individual.cpp Parameter.h Random.h Random.cpp)