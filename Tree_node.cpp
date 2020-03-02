//
// Created by LDNN97 on 2020/3/1.
//

#include "Tree_node.h"

using namespace std;

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

double node::cal(double l, double r, const double* xx){
    double ans = 0;
    if (symbol == "+") ans = l + r;
    if (symbol == "-") ans = l - r;
    if (symbol == "*") ans = l * r;
    if (symbol == "1") ans = xx[0];
    return ans;
}
