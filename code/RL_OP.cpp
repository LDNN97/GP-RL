//
// Created by LDNN97 on 2020/3/5.
//

#include "RL_OP.h"

namespace py = pybind11;
using namespace std;

void rl::env_reset(py::object &env, double* st){
    py::object st_or = env.attr("reset")();
    py::array_t<double, py::array::c_style | py::array::forcecast> st_tr(st_or);
    std::memcpy(st, st_tr.data(), st_tr.size() * sizeof(double));
}

void rl::env_step(pybind11::object &env, int &act, double* nst, double &reward, bool &end) {
    py::list nst_or = env.attr("step")(act);
//    py::print(nst_or);
    py::array_t<double, py::array::c_style | py::array::forcecast> nst_tr(nst_or[0]);
    std::memcpy(nst, nst_tr.data(), nst_tr.size() * sizeof(double));
    reward = py::cast<double>(nst_or[1]);
    end = py::cast<bool>(nst_or[2]);
//    py::print(nst_or[2]);
//    cout << "reset" << end  << endl;
}

rl::rec rl::get_max_action(py::object &env, individual* indi){
    double nst[n_observation]; double reward = 0; bool end = false;
    double max_v = 0; int max_act = 0; double v = 0;
    for (int i = 0; i <= n_action; i++){
        rl::env_step(env, i, nst, reward, end);
        v = reward + 1 * individual::calculate(indi->root, nst);
        if (v > max_v) {
            max_v = v;
            max_act = i;
        }
        env.attr("back_step")();
    }
    rl::rec ans{};
    ans.a = max_act; ans.v = max_v;
    return ans;
}

double rl::cal_target(pybind11::object &env, individual* lgbi){
    rec act = rl::get_max_action(env, lgbi);
    return act.v;
}

int rl::sample(const double * fitness){
    int ans = rand_int(0, POP_SIZE);
    for (int i = 0; i < T_S - 1; i++) {
        int tmp = rand_int(0, POP_SIZE);
        ans = (fitness[ans] > fitness[tmp]) ? ans : tmp;
    }
    return ans;
}

void rl::display() {
    cout << "Display the result" << endl;

    py::scoped_interpreter guard{};
    py::module::import("sys").attr("argv").attr("append")("");

    py::object gym = py::module::import("gym");
    py::object env = gym.attr("make")(env_name);

    individual* indi = individual::load_indi("Individual.txt");

    auto st = new double [n_observation]; auto nst = new double [n_observation];
    rl::rec action{}; double reward; bool end;

    env.attr("reset_ini")();
    rl::env_reset(env, st);
    double reward_indi = 0;
    for (int step = 0; step < 500; step++){
        action = rl::get_max_action(env, indi);
        rl::env_step(env, action.a, nst, reward, end);
        std::swap(st, nst);
        if (end) break;
        reward_indi += -1;
        env.attr("render")();
    }
    env.attr("close")();

    cout << "reward: " << reward_indi << endl;

    individual::clean(indi->root);
    delete indi;
}

void rl::rl_op() {
    // GYM
    py::scoped_interpreter guard{};
    py::module::import("sys").attr("argv").attr("append")("");

    py::object gym = py::module::import("gym");
    py::object env = gym.attr("make")(env_name);

    // build a model
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
    double fitness_total, reward_total;
    int best_indi;

    auto st = new double [n_observation]; auto nst = new double [n_observation];
    rec action{}; double reward; bool end;
    individual* lgbi;

    //Best individual in ini_pop
    env.attr("reset_ini")();
    best_indi = 0;
    memset(fitness, 0, sizeof(fitness));
    for (int i = 0; i < POP_SIZE; i++) {
        rl::env_reset(env, st);
        for (int step = 0; step < 500; step++){
            action = get_max_action(env, pop[i]);
            rl::env_step(env, action.a, nst, reward, end);
            if (end) break;
            fitness[i] += reward;
        }
        cout << i << " " << fitness[i] << endl;
        best_indi = (fitness[best_indi] > fitness[i]) ? best_indi : i;
    }
    cout << best_indi << endl;
    lgbi = new individual(*pop[best_indi]);

//    return;
    //Evolution
    for (int gen = 0; gen < MAX_GENERATION; gen++) {
        std::cout << "Generation: " << gen << std::endl;
        env.attr("reset_ini")();

        auto new_pop = new individual* [POP_SIZE];

        fitness_total = 0; reward_total = 0;
        memset(fitness, 0, sizeof(fitness));
        for (int i = 0; i < POP_SIZE; i++) {
            env_reset(env, st);
            int cnt = 0; double reward_indi = 0;
            for (int step = 0; step < 500; step++){
//                double target = rl::cal_target(env, lgbi);
//                double evolved = individual::calculate(pop[i]->root, st);
//                fitness[i] += (target - evolved) * (target - evolved);
//                cnt++;

                action = get_max_action(env, pop[i]);
                rl::env_step(env, action.a, nst, reward, end);
                std::swap(st, nst);
                if (end) break;
                reward_indi += -1;
            }
            fitness[i] = reward_indi;
//            fitness[i] /= double(cnt);
            fitness_total += fitness[i];
            reward_total += reward_indi;

            best_indi = (fitness[best_indi] > fitness[i]) ? best_indi : i;

            cout << i << " " << fitness[i] << " " << reward_indi << endl;
        }

//        delete lgbi;
//        lgbi = new individual(*pop[best_indi]);

        cout << " Average Fitness: " << fitness_total / double(POP_SIZE) << endl;
        cout << "Best ID and Fitness: " << best_indi << " " << fitness[best_indi] << endl;
        cout << "Tree Size: " << pop[best_indi]->root->size << endl;

        cout << "Average Reward: " << reward_total / double(POP_SIZE) << endl;

//        if ((fitness_total / double(POP_SIZE) < 1e-3) || (reward_total / double(POP_SIZE) > 450)) {
//            cout << "=====successfully!======" << endl;
//            cout << endl;
//            break;
//        }

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

    // save the model
    individual::save_indi(pop[best_indi]->root, "Individual.txt");

    // free pointer
    delete [] st;
    delete [] nst;

    for (int i = 0; i < POP_SIZE; i++) {
        individual::clean(pop[i]->root);
        delete pop[i];
    }
    delete [] pop;

}