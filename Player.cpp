//
// Created by LDNN97 on 2020/3/5.
//

// state -> action -> model
// model -> reward -> improve action
// sample many steps -> state + a => GP => fitness ~> evaluation  -> step sample action select the best

// interface reset(state_space, action_space), act(state) -> action, update(reward)

#include <iostream>
#include "Individual.h"
#include "Random.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
using namespace std;

int main() {


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

    // GYM
    py::scoped_interpreter guard{};
    py::object gym = py::module::import("gym");
    py::object env = gym.attr("make")("CartPole-v0");

    // for each individual: each episode contains 20 steps
    for (int i = 0; i < POP_SIZE; i++){
        py::object st_or = env.attr("reset")();
        py::array_t<double, py::array::c_style | py::array::forcecast> st_tr(st_or);
        std::vector<double> array_vec(st_tr.size());
        std::memcpy(array_vec.data(), st_tr.data(), st_tr.size() * sizeof(double));

        bool termi = false;
        for (int epi = 0; epi < 20; epi++){
            int action = rand_real(0, 1) < 0.5 ? 0 : 1;

        }
    }
}