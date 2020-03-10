//
// Created by LDNN97 on 2020/3/1.
//

#ifndef GP_CPP_INDIVIDUAL_H
#define GP_CPP_INDIVIDUAL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include "Parameter.h"
#include "Random.h"
#include "Tree_node.h"

typedef std::pair<int, std::string> item;

class individual {
public:
    node* root;

    explicit individual(node* pt = nullptr);
    individual(individual &indi);
    static node* tree_cpy(node* obj);
    static node* expand(const std::string &type, int depth, int max_depth);
    void build(const std::string &type, int max_depth);
    static double calculate(node* now, double* xx);
    static double cal_fitness(const individual& indi, int num,  double** xx, double* yy);
    static void show(node* now, std::vector<item>** pic, int* pos, int depth, int Max_depth);
    static void print_tree(individual* indi);
    static void in_order(node* now, node** seq, int &num);
    static void size_update(node* now);
    static int max_depth(node* now);
    void crossover(individual* another);
    static int cal_depth(node* now);
    static void mutation(individual* indi);

    // free pointer
    static void clean(node* now);

    // save model
    static void save_indi(node* now);
    static void load_indi(node* now);
};

#endif //GP_CPP_INDIVIDUAL_H
