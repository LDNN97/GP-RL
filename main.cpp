#include <iostream>
#include <cstdio>
#include <cstring>
#include <utility>
#include <random>
#include <set>

using namespace std;

const int MIN_DEPTH = 1;
const int MAX_DEPTH = 3;
const int MAX_GENERATION = 100;
const int T_S = 5; // Tournament Size
const double C_P = 0.8; // Crossover Probability
const double M_P = 0.2; // Mutation Probability
const int TYPE_NUM = 5;
const int POP_SIZE = (MAX_DEPTH - MIN_DEPTH + 1) * TYPE_NUM * 2;

// node
const int n_f = 3;
const string function_node[3]{"+", "-", "*"};
const int n_t = 1;
const string terminal_node[1]{"x0"};

// random generator
// random_device rd;
mt19937 mt(0); // seed()
uniform_real_distribution<double> dis(0, 1);

int rand_int(int lower, int upper) {
    return lower + int(dis(mt) * double(upper - lower));
}

class node{
public:
    node* father;
    node* left;
    node* right;
    int type;
    int size;
    string symbol;
    explicit node(node* fa = nullptr, node* l = nullptr, node* r = nullptr, int si = 1, int tp = 0, string sym = ""){
        father = fa;
        left = l;
        right = r;
        size = si;
        type = tp;
        symbol = std::move(sym); //To understand std::move
    }

    void set_symbol(int tp, string ty){
        type = tp;
        symbol = std::move(ty);
    }

    double cal(double l, double r, const double* xx){
        double ans = 0;
        if (symbol == "+") ans = l + r;
        if (symbol == "-") ans = l - r;
        if (symbol == "*") ans = l * r;
        if (symbol == "x0") ans = xx[0];
        return ans;
    }
};

class individual {
public:
    node* root;

    explicit individual(node* pt = nullptr){
        root = pt;
    }

    individual(individual &indi){
        root = tree_cpy(indi.root);
    }

    static node* tree_cpy(node* obj){
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

    static node* expand(const string &type, int depth, int max_depth){
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

    void build(const string &type, int max_depth){
        root = expand(type, 0, max_depth);
    }

    static double calculate(node* now, double* xx) {
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

    static double cal_fitness(const individual& indi, int num,  double** xx, double* yy){
        double fitness = 0;
        for (int i = 0; i < num; i++)
            fitness += abs(calculate(indi.root, xx[i]) - yy[i]);
        return fitness;
    }

    static void show(node* now){
        if (now == nullptr) return;
        cout << now << " " << now->symbol << " " << now->left << " " << now->right << endl;
        show(now->left);
        show(now->right);
    }

    static void in_order(node* now, node** seq, int &num) {
        if (now->left != nullptr) in_order(now->left, seq, num);
        seq[num++] = now;
        if (now->left != nullptr) in_order(now->right, seq, num);
    }

    static void size_update(node* now) {
        if (now == nullptr) return;
        now->size = now->left->size + now->right->size + 1;
        size_update(now->father);
    }

    void crossover(individual* another){
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

    static int cal_depth(node* now){
        if (now == nullptr) return 1;
        return cal_depth(now->father) + 1;
    }

    static void mutation(node* root){
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
    static void clean(node* now){
        if (now == nullptr) return;
        clean(now->left);
        clean(now->right);
        delete now;
    }
};

int sample(const double * fitness){
    int ans = rand_int(0, POP_SIZE);
    for (int i = 0; i < T_S - 1; i++) {
        int tmp = rand_int(0, POP_SIZE);
        ans = (fitness[ans] < fitness[tmp]) ? ans : tmp;
    }
    return ans;
}

int main() {
    freopen("train_data.txt", "r", stdin);
    int n, n_x;
    cin >> n >> n_x;
    auto train_x = new double* [n];
    auto train_y = new double [n];

    for (int i = 0; i < n; i++) {
        train_x[i] = new double [n_x];
        for (int j = 0; j < n_x; j++)
            cin >> train_x[i][j];
        cin >> train_y[i];
    }

    //Initialize
    auto pop = new individual* [POP_SIZE];
    for (int md = MIN_DEPTH; md <= MAX_DEPTH; md++) {
        for (int i = 0; i < TYPE_NUM; i++) {
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + i] = new individual();
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + i]->build("grow", md);
        }
        for (int i = 0; i < TYPE_NUM; i++) {
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + TYPE_NUM + i] = new individual();
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + TYPE_NUM + i]->build("full", md);
        }
    }

    //Display the tree
    for (int i = 0; i < POP_SIZE; i++) {
        cout << "ID: " << i << endl;
        individual::show(pop[i]->root);
    }

    //Calculate fitness
    double fitness[POP_SIZE];
    for (int i = 0; i < POP_SIZE; i++) {
        fitness[i] = individual::cal_fitness(*(pop[i]), n, train_x, train_y);
        cout << "ID: " << i << " " << fitness[i] << endl;
    }

    //Evolution
    for (int gen = 0; gen < MAX_GENERATION; gen++) {
        auto new_pop = new individual* [POP_SIZE];
        for (int i = 0; i < POP_SIZE; i++) {
            int index_p1 = sample(fitness);
            int index_p2 = sample(fitness);
            auto parent1 = new individual(*pop[index_p1]);
            auto parent2 = new individual(*pop[index_p2]);
            parent1->crossover(parent2);
            individual::mutation(parent1->root);
            new_pop[i] = parent1;
            individual::clean(parent2->root);
        }
        swap(pop, new_pop);

        // free pointer;
        for (int i = 0; i < POP_SIZE; i++)
            individual::clean(new_pop[i]->root);
        delete [] new_pop;
    }

    // free pointer
    for (int i = 0; i < POP_SIZE; i++)
        individual::clean(pop[i]->root);
    delete [] pop;
    delete [] train_y;
    for (int i = 0; i < n; i++)
        delete [] train_x[i];
    delete [] train_x;
}

