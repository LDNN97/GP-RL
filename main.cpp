#include <iostream>
#include "Parameter.h"
#include "Random.h"
#include "Individual.h"

using namespace std;

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

