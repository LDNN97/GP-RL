//
// Created by LDNN97 on 2020/4/2.
//

#ifndef GPRL_CPP_CARTPOLESWINGUP_H
#define GPRL_CPP_CARTPOLESWINGUP_H

#include "Random.h"
#include <array>
#include <armadillo>

class CartPoleSwingUp {
public:
    arma::vec state, state_ini, state_last;
    double t, t_last;

    CartPoleSwingUp();
    void reset_ini();
    void reset();
    void back_step();
    std::tuple<std::array<double, 4>, double, bool> step(double action);
};


#endif //GPRL_CPP_CARTPOLESWINGUP_H
