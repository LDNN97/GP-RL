#include <iostream>
#include "Parameter.h"
#include "Random.h"
#include "Individual.h"

using namespace std;

int sample(const double * fitness){
    int ans = rand_int(0, POP_SIZE);
    for (int i = 0; i < T_S - 1; i++) {
        int tmp = rand_int(0, POP_SIZE);

        cout << "compare: " << ans << " VS " << tmp << endl;
        cout << "fitness: " << fitness[ans] << " VS " << fitness[tmp] << endl;

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

    auto pic = new string * [MAX_DEPTH];
    for (int i = 0; i < MAX_DEPTH; i++)
        pic[i] = new string;

    // Display the tree
    for (int i = 0; i < POP_SIZE; i++) {
        if (i == 22)
            cout << "Break" << endl;
        cout << i << " " << pop[i] << endl;
        individual::print_tree(pop[i]);
    }

    //Calculate fitness
    double fitness[POP_SIZE];
    for (int i = 0; i < POP_SIZE; i++) {
        fitness[i] = individual::cal_fitness(*(pop[i]), n, train_x, train_y);
        cout << "ID: " << i << " " << fitness[i] << endl;
    }


    //Evolution
    for (int gen = 0; gen < MAX_GENERATION; gen++) {
        cout << "Generation: " << gen << endl;
        auto new_pop = new individual* [POP_SIZE];

        for (int i = 0; i < POP_SIZE; i++) {
            fitness[i] = individual::cal_fitness(*(pop[i]), n, train_x, train_y);
            cout << "ID: " << i << " " << fitness[i] << endl;
        }

        for (int i = 0; i < POP_SIZE; i++) {
            int index_p1 = sample(fitness);
            int index_p2 = sample(fitness);
            auto parent1 = new individual(*pop[index_p1]);
            auto parent2 = new individual(*pop[index_p2]);

            cout << "parent1 ID and address" << endl;
            cout << index_p1 << " " << pop[index_p1] << endl;
            individual::print_tree(pop[index_p1]);
            cout << "parent2 ID and address" << endl;
            cout << index_p2 << " " << pop[index_p2] << endl;
            individual::print_tree(pop[index_p2]);

            parent1->crossover(parent2);
//            individual::mutation(parent1->root);

            cout << "crossover result" << endl;
            cout << "parent1: " << parent1->root << endl;
            individual::print_tree(parent1);
            cout << "parent2: " << parent2->root << endl;
            individual::print_tree(parent2);

            new_pop[i] = parent1;
            individual::clean(parent2->root);
            delete parent2;
        }

//        cout << "old population" << endl;
//        for (int i = 0; i < POP_SIZE; i++)
//            cout << pop[i] << endl;
//        cout << "new population" << endl;
//        for (int i = 0; i < POP_SIZE; i++)
//            cout << new_pop[i] << endl;
        swap(pop, new_pop);
//        cout << "old population" << endl;
//        for (int i = 0; i < POP_SIZE; i++)
//            cout << pop[i] << endl;
//        cout << "new population" << endl;
//        for (int i = 0; i < POP_SIZE; i++)
//            cout << new_pop[i] << endl;

        // free pointer;
        for (int i = 0; i < POP_SIZE; i++) {
            individual::clean(new_pop[i]->root);
            delete new_pop[i];
        }
        delete [] new_pop;
    }

    for (int i = 0; i < POP_SIZE; i++) {
        cout << i << " " << pop[i] << endl;
        individual::print_tree(pop[i]);
    }

    // free pointer
    for (int i = 0; i < POP_SIZE; i++) {
        individual::clean(pop[i]->root);
        delete pop[i];
    }
    delete [] pop;

    delete [] train_y;

    for (int i = 0; i < n; i++)
        delete [] train_x[i];
    delete [] train_x;
}

