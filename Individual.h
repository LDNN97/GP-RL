//
// Created by LDNN97 on 2020/3/1.
//

#ifndef GP_CPP_INDIVIDUAL_H
#define GP_CPP_INDIVIDUAL_H

#include <iostream>
#include "Parameter.h"
#include "Random.h"
#include "Tree_node.h"


using namespace std;

class individual {
public:
    node* root;

    explicit individual(node* pt = nullptr);
    individual(individual &indi);
    static node* tree_cpy(node* obj);
    static node* expand(const string &type, int depth, int max_depth);
    void build(const string &type, int max_depth);
    static double calculate(node* now, double* xx);
    static double cal_fitness(const individual& indi, int num,  double** xx, double* yy);
    static void show(node* now);
    static void in_order(node* now, node** seq, int &num);
    static void size_update(node* now);
    void crossover(individual* another);
    static int cal_depth(node* now);
    static void mutation(node* root);
    // free pointer
    static void clean(node* now);
};

#endif //GP_CPP_INDIVIDUAL_H
