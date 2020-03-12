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
    double dist[POP_SIZE];
    double fitness_total, dist_total;
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

    std::array<double, MAX_GENERATION> f_a {};
    std::array<double, MAX_GENERATION> d_a {};
    //Evolution
    for (int gen = 0; gen < MAX_GENERATION; gen++) {
        std::cout << "Generation: " << gen << std::endl;
        env.attr("reset_ini")();

        auto new_pop = new individual* [POP_SIZE];

        best_indi = 0;fitness_total = 0; dist_total = 0;
        memset(fitness, 0, sizeof(fitness));
        memset(dist, 0, sizeof(dist));
        for (int i = 0; i < POP_SIZE; i++) {
            env_reset(env, st);
            int cnt = 0;
            for (int step = 0; step < 500; step++){
                double target = rl::cal_target(env, lgbi);
                double evolved = individual::calculate(pop[i]->root, st);
                dist[i] += (target - evolved) * (target - evolved);
                cnt++;

                action = get_max_action(env, pop[i]);
                rl::env_step(env, action.a, nst, reward, end);
                std::swap(st, nst);
                if (end) break;
                fitness[i] += -1;
            }
            fitness_total += fitness[i];
            dist[i] /= double(cnt);
            dist_total += dist[i];

            best_indi = (fitness[best_indi] > fitness[i]) ? best_indi : i;

            cout << i << " " << fitness[i] << " " << dist[i] << endl;
        }

        delete lgbi;
        lgbi = new individual(*pop[best_indi]);

        f_a[gen] = fitness_total / double(POP_SIZE);
        d_a[gen] = dist_total / double(POP_SIZE);
        cout << " Average Fitness and Dist " << f_a[gen] << " " << d_a[gen] << endl;
        cout << "Best ID and Fitness: " << best_indi << " " << fitness[best_indi] << endl;
        cout << "Tree Size: " << pop[best_indi]->root->size << endl;


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

    // print the average fit and dist
    ofstream file("Average.txt");
    for (int i = 0; i < MAX_GENERATION; i++)
        file << i << " " << f_a[i] << " " << d_a[i] << endl;
    file.close();

    // save the model
    individual::save_indi(pop[best_indi]->root, "Individual.txt");

    // print
    py::object _f_a = py::cast(f_a);
    py::object _d_a = py::cast(d_a);
    py::object plt = py::module::import("matplotlib.pyplot");
    plt.attr("figure")();

    plt.attr("subplot")(211);
    plt.attr("plot")(_f_a);
    plt.attr("subplot")(212);
    plt.attr("plot")(_d_a);
    plt.attr("yscale")("log");
    plt.attr("show")();

    // free pointer
    delete [] st;
    delete [] nst;

    for (int i = 0; i < POP_SIZE; i++) {
        individual::clean(pop[i]->root);
        delete pop[i];
    }
    delete [] pop;

}