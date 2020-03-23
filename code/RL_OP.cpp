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

void rl::env_step(pybind11::object &env, double &act, double* nst, double &reward, bool &end) {
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
    double max_v = 0; double max_act = 0; double v = 0;

    // Discrete
//    for (int i = 0; i <= n_action; i++){ // env: CartPole <  MountainCar <=
//        rl::env_step(env, i, nst, reward, end);
//        v = reward + 1 * individual::calculate(indi->root, nst);
//        if (v > max_v) {
//            max_v = v;
//            max_act = i;
//        }
//        env.attr("back_step")();
//    }

    // Continuous
    for (int i = 0; i <= 5; i++){ // env: CartPole <  MountainCar <=
        double act_tmp = rand_real(-1, 1);
        rl::env_step(env, act_tmp, nst, reward, end);
        v = reward + 1 * individual::calculate(indi->root, nst);
        if (v > max_v) {
            max_v = v;
            max_act = act_tmp;
        }
        env.attr("back_step")();
    }

    rl::rec ans{};
    ans.a = max_act; ans.v = max_v;
    return ans;
}

//double rl::cal_target(pybind11::object &env, individual* lgbi){
//    rec act = rl::get_max_action(env, lgbi);
//    return act.v;
//}

template <typename T>

vector<size_t> sort_indexes(const vector<T> &v) {
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}

void rl::get_rank(std::vector<double> &rank, std::vector<double> &fitness, std::vector<double> &dist, double fit_rate, double dis_rate ) {
    auto fit_index = sort_indexes(fitness);
    auto dis_index = sort_indexes(dist);
    std::array<double, POP_SIZE> fit_rk {};
    std::array<double, POP_SIZE> dis_rk {};
    for (int i = 0; i < POP_SIZE; i++) {
        fit_rk[fit_index[i]] = i;
        dis_rk[dis_index[i]] = i;
    }
    double _rank;
    cout << "rank" << endl;
    for (int i = 0; i < POP_SIZE; i++) {
        _rank = fit_rate * fit_rk[i] + dis_rate * dis_rk[i];
        rank.push_back(_rank);
        cout << i << " " << fit_rate << " " << dis_rate << " " << fit_rk[i] << " " << dis_rk[i] << " " << _rank <<  endl;
    }
}

int rl::sample(vector<double> &rank){
    int ans = rand_int(0, POP_SIZE);
    for (int i = 0; i < T_S - 1; i++) {
        int tmp = rand_int(0, POP_SIZE);
        ans = (rank[ans] > rank[tmp]) ? ans : tmp;
    }
    return ans;
}

void rl::display() {
    cout << "Display the result" << endl;

    py::scoped_interpreter guard{};
    py::module::import("sys").attr("argv").attr("append")("");

//    py::object gym = py::module::import("gym");
//    py::object env = gym.attr("make")(env_name);

    py::object env_list = py::module::import("env");
    py::object env = env_list.attr(env_name.c_str())();

    individual* indi = individual::load_indi("Individual.txt");

    auto st = new double [n_observation]; auto nst = new double [n_observation];
    rl::rec action{}; double reward; bool end;

    for (int i = 0; i < 5; i++) {
        env.attr("reset_ini")();
        rl::env_reset(env, st);
        double reward_indi = 0;
        for (int step = 0; step < 500; step++){
            action = rl::get_max_action(env, indi);
            rl::env_step(env, action.a, nst, reward, end);
            std::swap(st, nst);
            if (end) break;
            reward_indi += reward; //env CartPole +1 MountainCar -1
            env.attr("render")();
        }
        cout << reward_indi << endl;
    }
    env.attr("close")();

    individual::clean(indi->root);
    delete indi;
}

void rl::rl_op() {
    // GYM
    py::scoped_interpreter guard{};
    py::module::import("sys").attr("argv").attr("append")("");

    py::object env_list = py::module::import("env");
    py::object env = env_list.attr(env_name.c_str())();

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
    vector<double> fitness;
    vector<double> dist;
    double fitness_total, dist_total;
    int best_indi;

    auto st = new double [n_observation]; auto nst = new double [n_observation];
    rec action{}; double reward; bool end;
    individual* utnbi; double fitness_utnbi; double lgar; //last generation average reward
    vector<double> rank;

    //Best individual in ini_pop
    env.attr("reset_ini")();
    best_indi = 0;
    fitness.clear();
    for (int i = 0; i < POP_SIZE; i++) {
        rl::env_reset(env, st);
        double _fit_tot = 0;
        for (int step = 0; step < 500; step++){
            action = get_max_action(env, pop[i]);
            rl::env_step(env, action.a, nst, reward, end);
            if (end) break;
            _fit_tot += reward;
        }
        fitness.push_back(_fit_tot);
        cout << i << " " << fitness[i] << endl;
        best_indi = (fitness[best_indi] > fitness[i]) ? best_indi : i;
    }
    cout << best_indi << endl;
    utnbi = new individual(*pop[best_indi]);
    fitness_utnbi = fitness[best_indi];
    lgar = -1e6;

    std::array<double, MAX_GENERATION> b_i {};
    std::array<double, MAX_GENERATION> f_a {};
    std::array<double, MAX_GENERATION> d_a {};
    //Evolution
    for (int gen = 0; gen < MAX_GENERATION; gen++) {
        std::cout << "Generation: " << gen << std::endl;
        env.attr("reset_ini")();

        auto new_pop = new individual* [POP_SIZE];

        best_indi = 0; fitness_total = 0; dist_total = 0;
        fitness.clear();dist.clear();

        for (int i = 0; i < POP_SIZE; i++) {
            env_reset(env, st);
            int cnt = 0;
            double _dis_tot = 0;
            double _fit_tot = 0;
            for (int step = 0; step < 500; step++){
                action = get_max_action(env, pop[i]);

                rl::rec target = rl::get_max_action(env, utnbi);
                if (action.a == target.a)
                    _dis_tot += 1;
                cnt++;

                rl::env_step(env, action.a, nst, reward, end);
                std::swap(st, nst);
                if (end) break;
                _fit_tot += reward; //env CartPole +1 MountainCar -1
            }
            dist.push_back(_dis_tot / cnt);
            fitness.push_back(_fit_tot);

            dist_total += dist[i];
            fitness_total += fitness[i];

            best_indi = (fitness[best_indi] > fitness[i]) ? best_indi : i;

            cout << i << " " << fitness[i] << " " << dist[i] << endl;
        }

//        double fitness_lgbi = 0;
//        env_reset(env, st);
//        for (int step = 0; step < 500; step++){
//            action = get_max_action(env, lgbi);
//            rl::env_step(env, action.a, nst, reward, end);
//            std::swap(st, nst);
//            if (end) break;
//            fitness_lgbi += -1;
//        }

        f_a[gen] = fitness_total / double(POP_SIZE);
        d_a[gen] = dist_total / double(POP_SIZE);
        cout << " Average Fitness and Dist " << f_a[gen] << " " << d_a[gen] << endl;
        cout << "GBI: " << best_indi << " fitness: " << fitness[best_indi] <<
                                        " size: " << pop[best_indi]->root->size << endl;
        cout << "UTNBI: " << fitness_utnbi << endl;



        // select rank mode
        double fit_rate, dis_rate;

        // original
//        if (fitness[best_indi] >= fitness_utnbi) {
//            delete utnbi;
//            utnbi = new individual(*pop[best_indi]);
//            fitness_utnbi = fitness[best_indi];
//        }

        // method1
        if (f_a[gen] >= lgar) {
            if (fitness[best_indi] >= fitness_utnbi) {
                delete utnbi;
                utnbi = new individual(*pop[best_indi]);
                fitness_utnbi = fitness[best_indi];
                fit_rate = 1; dis_rate = 0;
            } else {
                fit_rate = 0.7; dis_rate = 0.3;
            }
            lgar = f_a[gen];
        } else {
            //method1
            if (fitness[best_indi] >= fitness_utnbi) {
                delete utnbi;
                utnbi = new individual(*pop[best_indi]);
                fitness_utnbi = fitness[best_indi];
                fit_rate = 0.3; dis_rate = 0.7;
            } else {
                fit_rate = 0; dis_rate = 1;
            }

            //method2
//            fit_rate = 0.5; dis_rate = 0.5;
//            if (fitness[best_indi] >= fitness_utnbi) {
//                delete utnbi;
//                utnbi = new individual(*pop[best_indi]);
//                fitness_utnbi = fitness[best_indi];
//            }

            lgar = f_a[gen];
        }

        b_i[gen] = fitness_utnbi;

        // get rank
        rank.clear();
        get_rank(rank, fitness, dist, fit_rate, dis_rate);

//        if ((fitness_total / double(POP_SIZE) < 1e-3) || (reward_total / double(POP_SIZE) > 450)) {
//            cout << "=====successfully!======" << endl;
//            cout << endl;
//            break;
//        }

        for (int i = 0; i < POP_SIZE; i++) {
            int index_p1 = sample(rank);
            int index_p2 = sample(rank);
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
        file << i << " " << b_i[i] << " " << f_a[i] << " " << d_a[i] << endl;
    file.close();

    // save the model
    individual::save_indi(utnbi->root, "Individual.txt");

    // print
    py::object _b_i = py::cast(b_i);
    py::object _f_a = py::cast(f_a);
    py::object _d_a = py::cast(d_a);
    py::object plt = py::module::import("matplotlib.pyplot");
    plt.attr("figure")();

    plt.attr("subplot")(311);
    plt.attr("plot")(_b_i);
    plt.attr("subplot")(312);
    plt.attr("plot")(_f_a);
    plt.attr("subplot")(313);
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
