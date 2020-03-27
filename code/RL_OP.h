//
// Created by LDNN97 on 2020/3/6.
//

#ifndef GPRL_CPP_RL_OP_H
#define GPRL_CPP_RL_OP_H

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

namespace rl{
    using indi::individual;
    typedef std::array<double, n_observation> state_arr;
    void env_reset(pybind11::object &env, state_arr &st);
    void env_step(pybind11::object &env, const int &act, state_arr &nst, double &reward, bool &end);
    int get_max_action(pybind11::object &env, individual* indi);
    double cal_target(pybind11::object &env, individual* lgbi);
    int sample(std::vector<double> &rank);
    void get_rank(std::vector<double> &rank, std::vector<double> &fitness, std::vector<double> &dist, double fit_rate, double dis_rate);
    void rl_op();
    void best_agent();
    void ensemble_agent();
}

#endif //GPRL_CPP_RL_OP_H
