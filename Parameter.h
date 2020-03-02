//
// Created by LDNN97 on 2020/3/1.
//

#ifndef GP_CPP_PARAMETER_H
#define GP_CPP_PARAMETER_H

#include <string>

const int MIN_DEPTH = 3;
const int INI_DEPTH = 5;
const int MUT_DEPTH = 3;
const int MAX_GENERATION = 1000;
const int T_S = 5; // Tournament Size
const double C_P = 0.5; // Crossover Probability
const double M_P = 0.2; // Mutation Probability
const int TYPE_NUM = 5;
const int POP_SIZE = (INI_DEPTH - MIN_DEPTH + 1) * TYPE_NUM * 2;

// node
const int n_f = 3;
const std::string function_node[3]{"+", "-", "*"};
const int n_t = 1;
const std::string terminal_node[1]{"1"};

#endif //GP_CPP_PARAMETER_H
