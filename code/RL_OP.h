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
#include <algorithm>
#include <queue>

namespace rl{
    using indi::individual;
    typedef std::array<double, n_observation> state_arr;

    typedef std::pair<individual*, double> agent_pair;
    struct cmp {
        bool operator()(const agent_pair &a1, const agent_pair &a2) const{
            return a1.second > a2.second;
        }
    };
    typedef std::priority_queue<agent_pair, std::vector<agent_pair>, cmp> pri_que;

    void env_reset(pybind11::object &env, state_arr &st);
    void env_step(pybind11::object &env, const int &act, state_arr &nst, double &reward, bool &end);
    int get_max_action(pybind11::object &env, individual* indi);
    double cal_target(pybind11::object &env, individual* lgbi);
    int sample(std::vector<double> &rank);
    void get_rank(std::vector<double> &rank, std::vector<double> &fitness, std::vector<double> &dist, double fit_rate, double dis_rate);
    void rl_op();
    void best_agent();
    void ensemble_agent();
    void agent_push(pri_que &agent, individual* indi, double fit);
    void agent_flat(pri_que &agent, std::vector<agent_pair> &agent_array);
}

#endif //GPRL_CPP_RL_OP_H
