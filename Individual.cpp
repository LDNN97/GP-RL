//
// Created by LDNN97 on 2020/3/1.
//

#include "Individual.h"

individual::individual(node* pt){
    root = pt;
}

individual::individual(individual &indi){
    root = tree_cpy(indi.root);
}

node* individual::tree_cpy(node* obj){
    node* now = new node(nullptr, nullptr, nullptr, obj->size, obj->type, obj->symbol);
    if (obj->left != nullptr) {
        now->left = tree_cpy(obj->left);
        now->left->father = now;
    }
    if (obj->right != nullptr) {
        now->right = tree_cpy(obj->right);
        now->right->father = now;
    }
    return now;
}

 node* individual::expand(const string &type, int depth, int max_depth){
    node* now = new node();
    if (depth < MIN_DEPTH || (depth < max_depth && type == "full")) {
        int tmp = rand_int(0, n_f);
        now->set_symbol(0, function_node[tmp]);
    } else if (depth == max_depth){
        int tmp = rand_int(0, n_t);
        now->set_symbol(1, terminal_node[tmp]);
        return now;
    } else {
        if (dis(mt) < 0.5) {
            int tmp = rand_int(0, n_f);
            now->set_symbol(0, function_node[tmp]);
        }else{
            int tmp = rand_int(0, n_t);
            now->set_symbol(1, terminal_node[tmp]);
            return now;
        }
    }
    now->left = expand(type, depth + 1, max_depth);
    now->left->father = now;
    now->size += (now->left)->size;
    now->right = expand(type,depth + 1, max_depth);
    now->right->father = now;
    now->size += (now->right)->size;
    return now;
}

void individual::build(const string &type, int max_depth){
    root = expand(type, 0, max_depth);
}

double individual::calculate(node* now, double* xx) {
    if (now == nullptr) return 0;
    if (now->type == 0) {
        double l_total = 0, r_total = 0;
        l_total = calculate(now->left, xx);
        r_total = calculate(now->right, xx);
        return now->cal(l_total, r_total, xx);
    } else {
        return now->cal(0, 0, xx);
    }
}

double individual::cal_fitness(const individual& indi, int num,  double** xx, double* yy){
    double fitness = 0;
    for (int i = 0; i < num; i++)
        fitness += abs(calculate(indi.root, xx[i]) - yy[i]);
    return fitness;
}

void individual::show(node* now){
    if (now == nullptr) return;
    cout << now << " " << now->symbol << " " << now->left << " " << now->right << endl;
    show(now->left);
    show(now->right);
}

void individual::in_order(node* now, node** seq, int &num) {
    if (now->left != nullptr) in_order(now->left, seq, num);
    seq[num++] = now;
    if (now->left != nullptr) in_order(now->right, seq, num);
}

void individual::size_update(node* now) {
    if (now == nullptr) return;
    now->size = now->left->size + now->right->size + 1;
    size_update(now->father);
}

void individual::crossover(individual* another){
    if (dis(mt) > C_P) return;

    int cnt = 0;
    auto seq1 = new node* [root->size];
    in_order(root, seq1, cnt);
    cnt = 0;
    auto seq2 = new node* [another->root->size];
    in_order(another->root, seq2, cnt);

    node* cross_node1 = seq1[rand_int(0, root->size)];
    node* cross_node2 = seq2[rand_int(0, another->root->size)];

    if (cross_node1->father->left == cross_node1)
        cross_node1->father->left = cross_node2;
    else
        cross_node1->father->right = cross_node2;

    if (cross_node2->father->left == cross_node2)
        cross_node2->father->left = cross_node1;
    else
        cross_node2->father->right = cross_node1;

    swap(cross_node1->father, cross_node2->father);

    size_update(cross_node1->father);
    size_update(cross_node2->father);

    delete [] seq1;
    delete [] seq2;
}

int individual::cal_depth(node* now){
    if (now == nullptr) return 1;
    return cal_depth(now->father) + 1;
}

void individual::mutation(node* root){
    if (dis(mt) > M_P) return;

    int cnt = 0;
    auto seq1 = new node* [root->size];
    in_order(root, seq1, cnt);
    node* mutation_node = seq1[rand_int(0, root->size)];

    int depth = cal_depth(mutation_node->father);

    string type;
    if (dis(mt) < 0.5)
        type = "grow";
    else
        type = "full";

    if (mutation_node->left == mutation_node)
        mutation_node->father->left = expand(type, depth, MAX_DEPTH);
    else
        mutation_node->father->right = expand(type, depth, MAX_DEPTH);

    size_update(mutation_node->father);

    clean(mutation_node);
}

// free pointer
void individual::clean(node* now){
    if (now == nullptr) return;
    clean(now->left);
    clean(now->right);
    delete now;
}

