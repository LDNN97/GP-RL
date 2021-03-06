//
// Created by LDNN97 on 2020/3/1.
//

#ifndef GPRL_CPP_PARAMETER_H
#define GPRL_CPP_PARAMETER_H

#include <string>
#include "Env.h"

const auto RAND_SEED = time(0); // 0 or time(0)
const int MIN_DEPTH = 3;
const int INI_DEPTH = 5;
const int MUT_DEPTH = 3;
const int MAX_GENERATION = 1000;
const int T_S = 6; // Tournament Size
const double C_P = 0.8; // Crossover Probability
const double M_P = 0.2; // Mutation Probability
const int TYPE_NUM = 5;
const int POP_SIZE = (INI_DEPTH - MIN_DEPTH + 1) * TYPE_NUM * 2;
const int ENSEMBLE_SIZE = T_S;

// !!!!!Remember to change the tree_node code about the terminal node!!!!!!

// GP Meta-Function Set
const int n_f = 4;
const std::string function_node[n_f]{"+", "-", "*", "/"};

//=====RL=====
//-----MountainCar-----
typedef MountainCar env_class;
const std::string env_name = "MountainCar";
const int n_observation = 2;
const int n_t = n_observation;
const std::string terminal_node[n_t]{"1", "2"};
const int n_action = 3;
const int action_set[n_action]{0, 1, 2};

//-----CartPole-----
//typedef CartPole env_class;
//const std::string env_name = "CartPole";
//const int n_observation = 4;
//const int n_t = n_observation;
//const std::string terminal_node[n_t]{"1", "2", "3", "4"};
//const int n_action = 2;
//const int action_set[n_action]{0, 1};

//-----CartPoleSwingUp-----
//typedef CartPoleSwingUp env_class;
//const std::string env_name = "CartPoleSwingUp";
//const int n_observation = 4;
//const int n_t = n_observation;
//const std::string terminal_node[n_t]{"1", "2", "3", "4"};
//const int n_action = 11;
//const double action_set[n_action]{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};

#endif //GPRL_CPP_PARAMETER_H
