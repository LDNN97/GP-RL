//
// Created by LDNN97 on 2020/3/6.
//
#include "Num_OP.h"

using namespace std;

int num::sample(const double * fitness){
    int ans = rand_int(0, POP_SIZE);
    for (int i = 0; i < T_S - 1; i++) {
        int tmp = rand_int(0, POP_SIZE);
        ans = (fitness[ans] < fitness[tmp]) ? ans : tmp;
    }
    return ans;
}

void num::num_op(){
    FILE* file = freopen("train_data.txt", "r", stdin);
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
    for (int md = MIN_DEPTH; md <= INI_DEPTH; md++) {
        for (int i = 0; i < TYPE_NUM; i++) {
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + i] = new individual();
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + i]->build("grow", md);
        }
        for (int i = 0; i < TYPE_NUM; i++) {
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + TYPE_NUM + i] = new individual();
            pop[TYPE_NUM * 2 * (md - MIN_DEPTH) + TYPE_NUM + i]->build("full", md);
        }
    }

    //fitness
    double fitness[POP_SIZE];
    double total;
    int best_indi;

    //Evolution
    for (int gen = 0; gen < MAX_GENERATION; gen++) {
        cout << "Generation: " << gen << endl;
        auto new_pop = new individual* [POP_SIZE];

        total = 0; best_indi = 0;
        for (int i = 0; i < POP_SIZE; i++) {
            fitness[i] = individual::cal_fitness(*(pop[i]), n, train_x, train_y);
            total += fitness[i];
            best_indi = (fitness[best_indi] < fitness[i]) ? best_indi : i;
        }
        cout << " Average Fitness: " << total / double(POP_SIZE) << endl;
        cout << "Best Fitness: " << fitness[best_indi] << endl;
        cout << "Tree Size: " << pop[best_indi]->root->size << endl;

        if (fitness[best_indi] < 1e-3) {
            cout << "=====successfully!======" << endl;
            cout << endl;
            break;
        }

        for (int i = 0; i < POP_SIZE; i++) {
            int index_p1 = sample(fitness);
            int index_p2 = sample(fitness);
            auto parent1 = new individual(*pop[index_p1]);
            auto parent2 = new individual(*pop[index_p2]);

            parent1->crossover(parent2);
            individual::mutation(parent1);

            new_pop[i] = parent1;
            individual::clean(parent2->root);
            delete parent2;
        }

        swap(pop, new_pop);

        // free pointer;
        for (int i = 0; i < POP_SIZE; i++) {
            individual::clean(new_pop[i]->root);
            delete new_pop[i];
        }
        delete [] new_pop;
    }

    cout << "Final Generation: " << endl;
    total = 0; best_indi = 0;
    for (int i = 0; i < POP_SIZE; i++) {
        fitness[i] = individual::cal_fitness(*(pop[i]), n, train_x, train_y);
        total += fitness[i];
        best_indi = (fitness[best_indi] < fitness[i]) ? best_indi : i;
    }
    cout << " Average Fitness: " << total / double(POP_SIZE) << endl;
    cout << "Best Fitness: " << fitness[best_indi] << endl;

    // test
    cin >> n >> n_x;
    cout << n << n_x;
    auto test_x = new double* [n];
    auto test_y = new double [n];

    for (int i = 0; i < n; i++) {
        test_x[i] = new double [n_x];
        for (int j = 0; j < n_x; j++)
            cin >> test_x[i][j];
        cin >> test_y[i];
        cout << individual::calculate(pop[best_indi]->root, test_x[i]) << " " << test_y[i] << endl;
    }

    delete [] test_y;
    for (int i = 0; i < n; i++)
        delete [] test_x[i];
    delete [] test_x;

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