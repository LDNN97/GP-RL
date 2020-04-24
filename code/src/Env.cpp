//
// Created by LDNN97 on 2020/4/2.
//

#include "../include/Env.h"

using namespace arma;

// MountainCar
MountainCar::MountainCar(const int seed) {
    arma_rng::set_seed(seed);
    double x = -0.6 + 0.2 * randu();
    state_ini = {x, 0};
    state = state_ini;
    state_last = state;
}

void MountainCar::reset_ini() {
    double x = -0.6 + 0.2 * randu();
    state_ini = {x, 0};
}

void MountainCar::reset() {
    state = state_ini;
}

void MountainCar::back_step() {
    state = state_last;
}

std::tuple<std::array<double, 2>, double, bool> MountainCar::step(double action) {
    double pos_min = -1.2, pos_max = 0.6, speed_max = 0.07, pos_goa = 0.5;
    double force = 0.001, gravity = 0.0025;

    state_last = state;
    double pos = state[0], vel = state[1];
    vel += (action - 1) * force + cos(3 * pos) * (-gravity);
    vel = vel < -speed_max ? -speed_max : vel;
    vel = vel > speed_max ? speed_max : vel;
    pos += vel;
    pos = pos < pos_min ? pos_min : pos;
    pos = pos > pos_max ? pos_max : pos;
    if (pos == pos_min && vel < 0)
        vel = 0;

    bool done = false;
    if (pos >= pos_goa && vel >= 0)
        done = true;

    double reward = -1;

    std::array<double, 2> obs {pos, vel};
    return std::tuple<std::array<double, 2>, double, bool>(obs, reward, done);
}

// CartPole
CartPole::CartPole(const int seed) {
    arma_rng::set_seed(seed);
    vec delta = {-0.05, -0.05, -0.05, -0.05};
    state_ini = delta + 0.1 * randu(4);
    state = state_ini;
    state_last = state;
    t = 0; t_last = 0;
}

void CartPole::reset_ini() {
    vec delta = {-0.05, -0.05, -0.05, -0.05};
    state_ini = delta + 0.1 * randu(4);
}

void CartPole::reset() {
    state = state_ini;
    t = 0; t_last = 0;
}

void CartPole::back_step() {
    state = state_last;
    t = t_last;
}

std::tuple<std::array<double, 4>, double, bool> CartPole::step(double action) {
    double g = 9.8, m_c = 1.0, m_p = 0.1, m_total = m_c + m_p;
    double l = 0.5, m_p_l = m_p * l, force_mag = 10, dt = 0.02;
    int t_limit = 1000;
    double x_threshold = 2.4, theta_threshold = 12 * 2 * datum::pi / 360;

    double force = -force_mag + 20 * action;

    state_last = state;
    double x = state[0], x_dot = state[1], theta = state[2], theta_dot = state[3];

    double s = sin(theta), c = cos(theta);
    double temp = (force + m_p_l * theta_dot * theta_dot * s) / m_total;
    double theta_a = (g * s - c * temp) / (l * (4.0/3.0 - m_p * c * c / m_total));
    double x_a = temp - m_p_l * theta_a * c / m_total;

    x += x_dot * dt;
    theta += theta_dot * dt;
    x_dot += x_a * dt;
    theta_dot += theta_a * dt;

    state = {x, x_dot, theta, theta_dot};

    bool done = false;
    if (x < -x_threshold || x > x_threshold ||
        theta < -theta_threshold || theta > theta_threshold)
        done = true;

    t_last = t; t += 1;
    if (t >= t_limit) done = true;

    double reward = done ? 0 : 1;

    std::array<double, 4> obs {x, x_dot, theta, theta_dot};
    return std::tuple(obs, reward, done);
}

// CartPoleSwingUp
CartPoleSwingUp::CartPoleSwingUp(const int seed) {
    arma_rng::set_seed(seed);
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
    double g = 9.82, m_c = 0.5, m_p = 0.5, m_total = m_c + m_p;
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

