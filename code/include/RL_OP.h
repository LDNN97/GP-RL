//
// Created by LDNN97 on 2020/3/6.
//

#ifndef GPRL_CPP_RL_OP_H
#define GPRL_CPP_RL_OP_H

#include "Individual.h"
#include "Random.h"
#include "Env.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <iostream>
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

    // Train
    int get_max_action(env_class &env, const individual* indi);
    int ensemble_selection(env_class &env, const std::vector<agent_pair> &agent);
    void rl_op(const std::string & _pre, const int seed, const std::string &method);

    // Display();
    void env_reset(pybind11::object &env, state_arr &st);
    void env_step(pybind11::object &env, const int &act, state_arr &nst, double &reward, bool &end);
    int ensemble_selection(pybind11::object &env, std::vector<individual*> &agent);
    void ensemble_agent(int ID, int ind);
}

#endif //GPRL_CPP_RL_OP_H
