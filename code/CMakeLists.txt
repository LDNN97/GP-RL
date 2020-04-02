cmake_minimum_required(VERSION 3.15)
project(GPRL_CPP)

set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(GPRL_CPP main.cpp Tree_node.h Individual.h Tree_node.cpp Individual.cpp Parameter.h Random.h Random.cpp RL_OP.cpp RL_OP.h Display.cpp)
target_link_libraries(GPRL_CPP PRIVATE pybind11::embed Threads::Threads)
