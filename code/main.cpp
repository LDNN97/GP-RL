#include "include/RL_OP.h"
#include "include/CartPoleSwingUp.h"

int main() {
    double succ_rate = 0;
    std::array<rl::result_item, MAX_GENERATION> result{};
    for (int i = 0; i < EXP_NUM; i++){
        std::cout << "EXP: " << i << std::endl;
        std::string _pre = "EXP ";
        std::string num = std::to_string(i);
        _pre.insert(4, num);
        _pre += " ";
        rl::rl_op(_pre, succ_rate, result);
    }
    std::ofstream _file("Result/Result.txt");
    for (int i = 0; i < MAX_GENERATION; i++)
        _file << i << " " << result[i].f_a / double(EXP_NUM) << " "
            << result[i].f_b / double(EXP_NUM) << " " << result[i].f_ens / double(EXP_NUM) << std::endl;
    _file.close();

//    rl::best_agent();
//    rl::ensemble_agent();
}

