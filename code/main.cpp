#include "include/RL_OP.h"
#include "include/Env.h"
#include <flags.h>

// todo: 1. model free
// todo: 2. restart.

int main(int argc, char *argv[]) {
    const flags::args args(argc, argv);

    const auto _mode = args.get<std::string>("mode");
    const auto _exp_id = args.get<int>("exp_id");
    const auto _seed = args.get<int>("seed");
    const auto _method = args.get<std::string>("method");
    const auto _indi = args.get<int>("indi");

    if (!_seed && !_mode) {
        std::cerr << "No seed or mode supplied. :(\n";
        return 1;
    }

    if (*_mode == "train") {
        int exp_id = !_exp_id ? 0 : *_exp_id;
        int seed = !_seed ? 0 : *_seed;
        std::string method = !_method ? "original" : *_method;

        std::string _pre = "EXP ";
        std::string num = std::to_string(exp_id);
        _pre.insert(4, num);
        _pre += " ";
        rl::rl_op(_pre, seed, method);

        return 0;
    }

    if (*_mode == "display") {
        int exp_id = !_exp_id ? 0 : *_exp_id;
        int indi = !_indi ? -1 : *_indi;

        rl::ensemble_agent(exp_id, indi);

        return 0;
    }

    return 0;
}

