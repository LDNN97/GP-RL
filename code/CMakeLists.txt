cmake_minimum_required(VERSION 3.10)
project(GPRL_CPP)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11)

find_package(spdlog)

find_package(Armadillo REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(GPRL_CPP main.cpp include/Tree_node.h include/Individual.h src/Tree_node.cpp src/Individual.cpp include/Parameter.h include/Random.h src/Random.cpp src/RL_OP.cpp include/RL_OP.h src/Display.cpp src/Env.cpp include/Env.h)
target_link_libraries(GPRL_CPP PRIVATE pybind11::embed ${ARMADILLO_LIBRARIES} Threads::Threads)
