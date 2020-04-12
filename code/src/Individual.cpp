//
// Created by LDNN97 on 2020/3/1.
//

#include "../include/Individual.h"

using namespace indi;
using std::memset;

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

 node* individual::expand(const std::string &type, int depth, int max_depth){
    node* now = new node();

    if (depth >= max_depth) {
        int tmp = rand_int(0, n_t);
        now->set_symbol(1, terminal_node[tmp]);
        return now;
    } else
        if (depth < MIN_DEPTH) {
            int tmp = rand_int(0, n_f);
            now->set_symbol(0, function_node[tmp]);
        } else {
            if (type == "grow") {
                if (rand_real(0, 1) < 0.5) {
                    int tmp = rand_int(0, n_f);
                    now->set_symbol(0, function_node[tmp]);
                }else{
                    int tmp = rand_int(0, n_t);
                    now->set_symbol(1, terminal_node[tmp]);
                    return now;
                }
            } else{
                int tmp = rand_int(0, n_f);
                now->set_symbol(0, function_node[tmp]);
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

void individual::build(const std::string &type, int max_depth){
    root = expand(type, 1, max_depth);
}

double individual::calculate(node* now, const t_arr &xx) {
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



void individual::show(node* now, std::vector<item>** pic, int* pos, int depth, int Max_depth){
    if (now->type == 1) {
        pic[depth - 1]->push_back(item(pos[depth - 1], now->symbol));
        int number = 1 << (Max_depth - depth);
        for (int i = depth - 1; i < Max_depth; i++)
            pos[i] += number;
        return;
    }

    pic[depth - 1]->push_back(item(pos[depth - 1], now->symbol));
    int number = 1 << (Max_depth - depth);
    pos[depth - 1] += number;

    show(now->left, pic, pos, depth + 1, Max_depth);
    show(now->right, pic, pos, depth + 1, Max_depth);
}

int individual::max_depth(node* now) {
    if (now == nullptr) return 0;
    int l_depth = max_depth(now->left);
    int r_depth = max_depth(now->right);
    return std::max(l_depth, r_depth) + 1;
}

void individual::print_tree(individual* indi){
    int Max_depth = max_depth(indi->root);

    int pos[Max_depth];
    memset(pos, 0, sizeof(pos));
    auto pic = new std::vector<item>* [Max_depth];
    for (int i = 0; i < Max_depth; i++)
        pic[i] = new std::vector<item>;

    individual::show(indi->root, pic, pos, 1, Max_depth);

    for (int i = 0; i < Max_depth; i++) {
        if (pic[i]->size() == 1) {
            if ((*pic[i])[0].first > 0)
                for (int k = 0; k < (*pic[i])[0].first; k++) std::cout << " ";
            std::cout << (*pic[i])[0].second << std::endl;
        } else{
            int last = int(pic[i]->size()) - 1;
            for (int j = 0; j < last; j++) {
                if (j == 0 && (*pic[i])[j].first > 0)
                    for (int k = 0; k < (*pic[i])[j].first; k++) std::cout << " ";
                std::cout << (*pic[i])[j].second;
                int whitespace_number = (*pic[i])[j + 1].first - (*pic[i])[j].first - 1;
                for (int k = 0; k < whitespace_number; k++) std::cout << " ";
            }
            std::cout << (*pic[i])[last].second << std::endl;
        }

    }

    for (int i = 0; i < Max_depth; i++)
        delete pic[i];
    delete [] pic;
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
    if (rand_real(0, 1) > C_P) return;

//    cout << "Crossover!" << endl;

    int cnt = 0;
    auto seq1 = new node* [root->size];
    in_order(root, seq1, cnt);
    cnt = 0;
    auto seq2 = new node* [another->root->size];
    in_order(another->root, seq2, cnt);

//    cout << "parent1: " << endl;
//    for (int i = 0; i < root->size; i++)
//        cout << i << " " << seq1[i] << endl;
//    cout << "parent2: " << endl;
//    for (int i = 0; i < another->root->size; i++)
//        cout << i << " " << seq2[i] << endl;

    node* cross_node1 = seq1[rand_int(0, root->size)];
    node* cross_node2 = seq2[rand_int(0, another->root->size)];

//    cout << "parent1 crossover point: " << endl;
//    cout << cross_node1 << endl;
//    cout << "parent2 crossover point: " << endl;
//    cout << cross_node2 << endl;

    if (cross_node1->father == nullptr && cross_node2->father == nullptr) {
        std::swap(cross_node1, cross_node2);
        std::swap(root, another->root);
    }else if (cross_node1->father == nullptr) {
        if (cross_node2->father->left == cross_node2)
            cross_node2->father->left = cross_node1;
        else
            cross_node2->father->right = cross_node1;
        cross_node1->father = cross_node2->father;
        cross_node2->father = nullptr;
        root = cross_node2;
    }else if (cross_node2->father == nullptr) {
        if (cross_node1->father->left == cross_node1)
            cross_node1->father->left = cross_node2;
        else
            cross_node1->father->right = cross_node2;
        cross_node2->father = cross_node1->father;
        cross_node1->father = nullptr;
        another->root = cross_node1;
    }else {
        if (cross_node1->father->left == cross_node1)
            cross_node1->father->left = cross_node2;
        else
            cross_node1->father->right = cross_node2;
        if (cross_node2->father->left == cross_node2)
            cross_node2->father->left = cross_node1;
        else
            cross_node2->father->right = cross_node1;
        std::swap(cross_node1->father, cross_node2->father);
    }

//    cout << "parent1 tree size and cross_node1 subtree size" << endl;
//    cout << root->size << " " << cross_node1->size << endl;
//    cout << "parent2 tree size and cross_node2 subtree size" << endl;
//    cout << another->root->size << " " << cross_node2->size << endl;

    size_update(cross_node1->father);
    size_update(cross_node2->father);

//    cout << "parent1 tree size: " << root->size << endl;
//    cout << "parent2 tree size: " << another->root->size << endl;

    delete [] seq1;
    delete [] seq2;
}

int individual::cal_depth(node* now){
    if (now == nullptr) return 0;
    return cal_depth(now->father) + 1;
}

void individual::mutation(individual* indi){
    if (rand_real(0, 1) > M_P) return;

    int cnt = 0;
    auto seq1 = new node* [indi->root->size];
    in_order(indi->root, seq1, cnt);
    node* mutation_node = seq1[rand_int(0, indi->root->size)];

    int depth = cal_depth(mutation_node);
//    int sub_tree_depth = max_depth(mutation_node);
    int sub_tree_depth = MUT_DEPTH;

//    cout << depth << " " << sub_tree_depth << endl;

    std::string type;
    if (rand_real(0, 1) < 0.9)
        type = "grow";
    else
        type = "full";

    if (mutation_node->father == nullptr){
        indi->root = expand(type, depth, depth + sub_tree_depth - 1);
        tree_clean(mutation_node);
    }else {
        if (mutation_node->father->left == mutation_node)
            mutation_node->father->left = expand(type, depth, depth + sub_tree_depth - 1);
        else
            mutation_node->father->right = expand(type, depth, depth + sub_tree_depth - 1);
        size_update(mutation_node->father);
        tree_clean(mutation_node);
    }

    delete [] seq1;
}

// free pointer
void individual::tree_clean(node* now){
    if (now == nullptr) return;
    tree_clean(now->left);
    tree_clean(now->right);
    delete now;
}

void individual::indi_clean(individual* indi){
    tree_clean(indi->root);
    delete indi;
}

typedef std::pair<int, int> rela;
typedef std::pair<int, rela> link;

void individual::save_indi(node* now, const std::string& file_name){
    std::ofstream file(file_name);
    file << now->size << std::endl;

    std::vector<link> tree;
    node* que[now->size + 5];
    int f_ptr = 0, b_ptr = 1;
    que[0] = now;

    file << 0 << " " <<
        que[0]->size << " " << que[0]->type << " " << que[0]->symbol << std::endl;

    while (f_ptr < b_ptr){
        if (que[f_ptr]->left != nullptr) {
            tree.emplace_back(f_ptr, rela(0, b_ptr));
            que[b_ptr] = que[f_ptr]->left;

            file << b_ptr << " " <<
                que[b_ptr]->size << " " << que[b_ptr]->type << " " << que[b_ptr]->symbol << std::endl;

            b_ptr += 1;
        }
        if (que[f_ptr]->right != nullptr) {
            tree.emplace_back(f_ptr, rela(1, b_ptr));
            que[b_ptr] = que[f_ptr]->right;

            file << b_ptr << " " <<
                que[b_ptr]->size << " " << que[b_ptr]->type << " " << que[b_ptr]->symbol << std::endl;

            b_ptr += 1;
        }
        f_ptr++;
    }

    for (auto item : tree)
        file << item.first << " " << item.second.first << " " << item.second.second << std::endl;

    file.close();
}

individual* individual::load_indi(const std::string& file_name){
    int size;
    std::ifstream file(file_name);
    file >> size;

    node* nodes[size];
    int id, si, ty; std::string str;
    for (int i = 0; i < size; i++) {
        file >> id >> si >> ty >> str;
        node* now = new node(nullptr, nullptr, nullptr, si, ty, str);
        nodes[id] = now;
    }

    int parent, lor, son;
    for (int i = 0; i < size - 1; i++){
        file >> parent >> lor >> son;
        if (lor == 0) {
            nodes[parent]->left = nodes[son];
            nodes[son]->father = nodes[parent];
        } else {
            nodes[parent]->right = nodes[son];
            nodes[son]->father = nodes[parent];
        }
    }

    file.close();
    auto* indi = new individual(nodes[0]);
    return indi;
}



