//
// Created by LDNN97 on 2020/4/2.
//

#ifndef GPRL_CPP_ENV_H
#define GPRL_CPP_ENV_H

#include "Random.h"
#include <array>
#include <armadillo>

class MountainCar {
public:
    arma::vec state, state_ini, state_last;

    MountainCar(const int seed = 0);
    void reset_ini();
    void reset();
    void back_step();
    std::tuple<std::array<double, 2>, double, bool> step(double action);
};

class CartPole {
public:
    arma::vec state, state_ini, state_last;
    double t, t_last;

    CartPole(const int seed = 0);
    void reset_ini();
    void reset();
    void back_step();
    std::tuple<std::array<double, 4>, double, bool> step(double action);
};

class CartPoleSwingUp {
public:
    arma::vec state, state_ini, state_last;
    double t, t_last;

    CartPoleSwingUp(const int seed = 0);
    void reset_ini();
    void reset();
    void back_step();
    std::tuple<std::array<double, 4>, double, bool> step(double action);
};

#endif //GPRL_CPP_ENV_H
