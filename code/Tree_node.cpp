//
// Created by LDNN97 on 2020/3/1.
//

#include "Tree_node.h"

using namespace tree_node;
using std::string;

node::node(node* fa, node* l, node* r, int si, int tp, string sym){
    father = fa;
    left = l;
    right = r;
    size = si;
    type = tp;
    symbol = std::move(sym); //To understand std::move
}

void node::set_symbol(int tp, string ty){
    type = tp;
    symbol = std::move(ty);
}

double node::cal(double l, double r, const t_arr &xx){
    double ans = 0;
    if (symbol == "+") ans = l + r;
    if (symbol == "-") ans = l - r;
    if (symbol == "*") ans = l * r;
    if (symbol == "/") ans = (abs(r) < 0.001) ? l : l/r;
    if (symbol == "1") ans = xx[0];
    if (symbol == "2") ans = xx[1];
    if (symbol == "3") ans = xx[2];
    if (symbol == "4") ans = xx[3];
    return ans;
}
