//
// Created by LDNN97 on 2020/3/5.
//

// state -> action -> model
// model -> reward -> improve action
// sample many steps -> state + a => GP => fitness ~> evaluation  -> step sample action select the best

// interface reset(state_space, action_space), act(state) -> action, update(reward)

#include "RL_OP.h"

namespace py = pybind11;

void rl::env_reset(py::object &env, double* st){
    py::object st_or = env.attr("reset")();
    py::array_t<double, py::array::c_style | py::array::forcecast> st_tr(st_or);
    std::memcpy(st, st_tr.data(), st_tr.size() * sizeof(double));
}

void rl::env_step(pybind11::object &env, int &act, double* nst, double &reward, bool &end) {
    py::list nst_or = env.attr("step")(act);
    py::array_t<double, py::array::c_style | py::array::forcecast> nst_tr(nst_or[0]);
    std::memcpy(nst, nst_tr.data(), nst_tr.size() * sizeof(double));
    reward = py::cast<double>(nst_or[1]);
    end = py::cast<bool>(nst_or[2]);
}

int rl::sample(const double * fitness){
    int ans = rand_int(0, POP_SIZE);
    for (int i = 0; i < T_S - 1; i++) {
        int tmp = rand_int(0, POP_SIZE);
        ans = (fitness[ans] < fitness[tmp]) ? ans : tmp;
    }
    return ans;
}

int rl::get_max_action(double* st){

}

void rl::rl_op() {
    // GYM
    py::scoped_interpreter guard{};
    py::object gym = py::module::import("gym");
    py::object env = gym.attr("make")("CartPole-v0");

    // build a model
    auto pop = new individual* [POP_SIZE];
    for (int md = MIN_DEPTH; md <= INI_DEPTH; md++) {
        for (int i = 0; i < TYPE_NUM; i++) {
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + i] = new individual();
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + i]->build("grow", md);
        }
        for (int i = 0; i < TYPE_NUM; i++) {
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + TYPE_NUM + i] = new individual();
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + TYPE_NUM + i]->build("full", md);
        }
    }

    //fitness
    double fitness[POP_SIZE];
    double total;
    int best_indi;

    auto ini_st = new double [4]; auto st = new double [4]; auto nst = new double [4];
    double reward; bool end;
    individual** old_pop;

    //Todo: 1. best individual in ini_pop. 2. crossover and mutation

    for (int gen = 0; gen < MAX_GENERATION; gen ++) {
        old_pop = pop;
        rl::env_reset(env, ini_st);
        for (int i = 0; i < POP_SIZE; i++) {
            //Todo: computate value using gp_tree in the last generation V_target(x) = r + a * V'(f(x, a))
            std::memcpy(st, ini_st, sizeof(*ini_st));  // ?
            for (int step = 0; step < 20; step++){
                // epsilon-greedy
                int action = get_max_action(st);
                action = (rand_real(0, 1) < 0.2) ? rand_int(0, 2) : action;
                rl::env_step(env, action, nst, reward, end);
                std::swap(st, nst);
            }
            //Todo: fitness = 1/20 * sum((V_target(x) - V_eolved(x))^2)
        }
    }

}