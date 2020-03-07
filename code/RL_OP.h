//
// Created by LDNN97 on 2020/3/6.
//

#ifndef GP_CPP_RL_OP_H
#define GP_CPP_RL_OP_H

#include "Individual.h"
#include "Random.h"

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace rl{
    struct rec{
        int a;
        double v;
    };

    void env_reset(pybind11::object &env, double* st);
    void env_step(pybind11::object &env, int &act, double* nst, double &reward, bool &end);
    rec get_max_action(pybind11::object &env, individual* indi);
    double cal_target(pybind11::object &env, individual* lgbi);
    int sample(const double * fitness);

    void rl_op();
}

#endif //GP_CPP_RL_OP_H
