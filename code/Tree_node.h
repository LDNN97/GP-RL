//
// Created by LDNN97 on 2020/3/1.
//

#ifndef GP_CPP_TREE_NODE_H
#define GP_CPP_TREE_NODE_H

#include <string>
#include <cmath>

class node{
public:
    node* father;
    node* left;
    node* right;
    int type;
    int size;
    std::string symbol;
    explicit node(node* fa = nullptr, node* l = nullptr, node* r = nullptr, int si = 1, int tp = 1, std::string sym = "");
    void set_symbol(int tp, std::string ty);
    double cal(double l, double r, const double* xx);
};

#endif //GP_CPP_TREE_NODE_H
