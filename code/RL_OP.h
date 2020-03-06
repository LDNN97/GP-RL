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
    void env_reset(pybind11::object &env, double* st);
    void env_step(pybind11::object &env, int &act, double* nst, double &reward, bool &end);
    int sample(const double * fitness);
    int get_max_action(double* st);
    void rl_op();
}

#endif //GP_CPP_RL_OP_H
