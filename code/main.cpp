#include "include/RL_OP.h"
#include "include/CartPoleSwingUp.h"

// todo: 1. According to the similarity, random choose an indi from ensemble agents to replace fit_utnb.
// todo: 2. restart.
// todo: 3. model free

int main() {
    // Training
    double succ_rate = 0;
    std::array<rl::result_item, MAX_GENERATION> result{};
    for (int i = 0; i < EXP_NUM; i++){
        std::cout << "EXP: " << i << std::endl;
        std::string _pre = "EXP ";
        std::string num = std::to_string(i);
        _pre.insert(4, num);
        _pre += " ";
        rl::rl_op(i, _pre, succ_rate, result);
    }
    std::ofstream _file("Result/Result.txt");
    for (int i = 0; i < MAX_GENERATION; i++)
        _file << i << " " << result[i].f_a / double(EXP_NUM) << " "
            << result[i].f_b / double(EXP_NUM) << " " << result[i].f_ens / double(EXP_NUM) << std::endl;
    _file.close();

    // Display: choose !only! one of the following agents
//    rl::best_agent(0);
//    rl::ensemble_agent(0);
//
//    std::vector<int> exp_set; // push the EXP_ID in exp_set
//    exp_set.emplace_back(0);
//    exp_set.emplace_back(1);
//    rl::population_agent(exp_set);
}

