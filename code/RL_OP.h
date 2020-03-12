//
// Created by LDNN97 on 2020/3/6.
//

#ifndef GP_CPP_RL_OP_H
#define GP_CPP_RL_OP_H

#include "Individual.h"
#include "Random.h"

#include <array>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <numeric>      // std::iota
#include <algorithm>

const std::string env_name = "MountainCar-v0"; // CartPole-v0 or MountainCar-v0
const int n_action = 2;
const int n_observation = 2;

namespace rl{
    struct rec{
        int a;
        double v;
    };

    void env_reset(pybind11::object &env, double* st);
    void env_step(pybind11::object &env, int &act, double* nst, double &reward, bool &end);
    rec get_max_action(pybind11::object &env, individual* indi);
    double cal_target(pybind11::object &env, individual* lgbi);
    int sample(std::vector<double> &rank);
    std::vector<double> get_rank(std::vector<double> &fitness, std::vector<double> &dist, double fit_rate, double dis_rate);
    void rl_op();
    void display();
}

#endif //GP_CPP_RL_OP_H
