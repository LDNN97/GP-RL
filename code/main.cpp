#include "RL_OP.h"

int main() {
    pybind11::scoped_interpreter guard{};
    pybind11::module::import("sys").attr("argv").attr("append")("");

    rl::rl_op();
//    rl::best_agent();
//    rl::ensemble_agent();

}

