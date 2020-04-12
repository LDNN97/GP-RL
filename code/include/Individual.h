//
// Created by LDNN97 on 2020/3/1.
//

#ifndef GPRL_CPP_INDIVIDUAL_H
#define GPRL_CPP_INDIVIDUAL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <array>
#include <cstring>
#include "Parameter.h"
#include "Random.h"
#include "Tree_node.h"

namespace indi{
    typedef std::pair<int, std::string> item;
    typedef std::array<double, n_t> t_arr;

    using tree_node::node;

    class individual {
    public:
        node* root;

        explicit individual(node* pt = nullptr);
        individual(individual &indi);
        static node* tree_cpy(node* obj);
        static node* expand(const std::string &type, int depth, int max_depth);
        void build(const std::string &type, int max_depth);
        static double calculate(node* now, const t_arr &xx);
        static void show(node* now, std::vector<item>** pic, int* pos, int depth, int Max_depth);
        static void print_tree(individual* indi);
        static void in_order(node* now, node** seq, int &num);
        static void size_update(node* now);
        static int max_depth(node* now);
        void crossover(individual* another);
        static int cal_depth(node* now);
        static void mutation(individual* indi);

        // free pointer
        static void tree_clean(node* now);
        static void indi_clean(individual* indi);

        // save model
        static void save_indi(node* now, const std::string& file_name);
        static individual* load_indi(const std::string& file_name);
    };
}


#endif //GPRL_CPP_INDIVIDUAL_H
