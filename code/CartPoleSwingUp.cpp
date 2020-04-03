//
// Created by LDNN97 on 2020/4/2.
//

#include "CartPoleSwingUp.h"

using namespace arma;

CartPoleSwingUp::CartPoleSwingUp() {
    arma_rng::set_seed(0);
    state_ini = {0, 0, datum::pi, 0};
    vec delta = {-0.2, -0.2, -0.2, -0.2};
    state_ini += (delta + 0.4 * randu(4));
    state = state_ini;
    state_last = state;
    t = 0; t_last = 0;
}

void CartPoleSwingUp::reset_ini() {
    state_ini = {0, 0, datum::pi, 0};
    vec delta = {-0.2, -0.2, -0.2, -0.2};
    state_ini += (delta + 0.4 * randu(4));
}

void CartPoleSwingUp::reset() {
    state = state_ini;
    t = 0; t_last = 0;
}

void CartPoleSwingUp::back_step() {
    state = state_last;
    t = t_last;
}

std::tuple<std::array<double, 4>, double, bool> CartPoleSwingUp::step(double action) {
    double g = 9.82, m_c = 0.5, m_p = 0.5, m_total = 1;
    double l = 0.6, m_p_l = m_p * l, force_mag = 10, dt = 0.01, b = 0.1;
    int t_limit = 1000;
    double x_threshold = 2.4;

    double force = -force_mag + 20 * action;
    state_last = state;

    double x = state[0], x_dot = state[1], theta = state[2], theta_dot = state[3];
    double s = sin(theta), c = cos(theta);
    double x_d_numer = -2 * m_p_l * theta_dot * theta_dot * s +
                    3 * m_p * g * s * c + 4 * force - 4 * b * x_dot;
    double x_d_denom = 4 * m_total - 3 * m_p * c * c;
    double xdot_update = x_d_numer / x_d_denom;
    double t_d_numer = -3 * m_p_l * theta_dot * theta_dot * s *c +
                    6 * m_total * g * s + 6 * (force - b * x_dot) * c;
    double t_d_denom = 4 * l * m_total - 3 * m_p * l * c * c;
    double tdot_update = t_d_numer / t_d_denom;

    x += x_dot * dt;
    theta += theta_dot * dt;
    x_dot += xdot_update * dt;
    theta_dot += tdot_update * dt;

    state = {x, x_dot, theta, theta_dot};

    bool done = false;
    if (x < -x_threshold || x > x_threshold) done = true;
    t_last = t; t += 1;
    if (t >= t_limit) done = true;

    double reward_theta = cos(theta), reward_x = cos((x / x_threshold) * (datum::pi / 2));
    double reward = reward_theta * reward_x;

    std::array<double, 4> obs {x, x_dot, theta, theta_dot};
    return std::tuple(obs, reward, done);
}







