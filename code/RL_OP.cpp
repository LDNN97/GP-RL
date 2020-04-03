//
// Created by LDNN97 on 2020/3/5.
//


#include "RL_OP.h"
#include "spdlog/spdlog.h"
#include "taskflow/taskflow.hpp"

namespace py = pybind11;
using namespace rl;
using std::endl;
using std::vector;
using std::swap;
using std::ofstream;
using std::ifstream;
using std::string;
using std::priority_queue;

int rl::get_max_action(CartPoleSwingUp &env, const individual* indi){
    state_arr nst{};
    double max_v = -1e6; int max_act = 0; double v;

    for (int i = 0; i < n_action; i++){
        auto [nst, reward, end] = env.step(action_set[i]);
        v = reward + 1 * individual::calculate(indi->root, nst);
        if (v > max_v) {
            max_v = v;
            max_act = i;
        }
        env.back_step();
    }

    return max_act;
}

int rl::ensemble_selection(CartPoleSwingUp &env, vector<agent_pair> &agent_array) {
    std::array<int, n_action> box{};

    for (int i = 0; i < agent_array.size(); i++) {
        state_arr nst{};
        double max_v = -1e6; int max_act = 0; double v;
        for (double j : action_set){
            auto [nst, reward, end] = env.step(j);
            v = reward + 1 * individual::calculate(agent_array[i].first->root, nst);
            if (v > max_v) {
                max_v = v;
                max_act = i;
            }
            env.back_step();
        }
        box[max_act]++;
    }

    int ans = std::distance(box.begin(), std::max_element(box.begin(), box.end()));

    return ans;
}

void evaluate(CartPoleSwingUp env, const individual* indi, vector<agent_pair> &agent, double &fit, double &sim){
    std::array<double, n_observation> nst{};
    int action; int action_target;

    env.reset();
    int cnt = 0;
    double _fit = 0, _sim = 0;
    for (int step = 0; step < 1000; step++){
        cnt++;
        action = get_max_action(env, indi);

        action_target = rl::ensemble_selection(env, agent);
        if (action == action_target)
            _sim += 1;

        auto [nst, reward, end] = env.step(action_set[action]);
        if (end) break;
        _fit += reward;
    }
    _sim /= double(cnt);

    fit = _fit; sim = _sim;
}

void agent_flat(pri_que &agent, vector<agent_pair> &agent_array){
    while (!agent.empty()){
        agent_array.emplace_back(agent.top());
        agent.pop();
    }
    for (auto indi : agent_array) agent.push(indi);
}

void agent_push(pri_que &agent, individual* indi, double fit){
    if (agent.empty()) {
        auto tmp = new individual(*indi);
        agent.push(agent_pair(tmp, fit));
    }
    else {
        agent_pair top = agent.top();
        if (top.second <= fit) {
            auto tmp = new individual(*indi);
            agent.push(agent_pair(tmp, fit));
        }
        if (agent.size() > ENSEMBLE_SIZE) {
            auto tmp = agent.top();
            individual::indi_clean(tmp.first);
            agent.pop();
        }
    }
}

// Rank
template <typename T>

vector<size_t> sort_indexes(const std::array<T, POP_SIZE> &v) {
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}

void get_rank(std::vector<double> &rank, std::array<double, POP_SIZE> &fit, std::array<double, POP_SIZE> &sim,
                  double fit_rate, double sim_rate) {
    auto fit_index = sort_indexes(fit);
    auto dis_index = sort_indexes(sim);
    std::array<double, POP_SIZE> fit_rk {};
    std::array<double, POP_SIZE> sim_rk {};
    for (int i = 0; i < POP_SIZE; i++) {
        fit_rk[fit_index[i]] = i;
        sim_rk[dis_index[i]] = i;
    }
    double _rank;
    for (int i = 0; i < POP_SIZE; i++) {
        _rank = fit_rate * fit_rk[i] + sim_rate * sim_rk[i];
        rank.push_back(_rank);
    }
}

// Tournament Selection
int sample(vector<double> &rank){
    int ans = rand_int(0, POP_SIZE);
    for (int i = 0; i < T_S - 1; i++) {
        int tmp = rand_int(0, POP_SIZE);
        ans = (rank[ans] > rank[tmp]) ? ans : tmp;
    }
    return ans;
}

void model_save(individual* &indi_utnb, pri_que &agent){
    // save best model
    individual::save_indi(indi_utnb->root, "Agent/best_agent.txt");

    // agent flat
    vector<agent_pair> agent_array;
    agent_flat(agent, agent_array);

    ofstream _file("Agent/ensemble_size.txt");
    _file << agent_array.size() << endl;
    _file.close();

    for (int i = 0; i < agent_array.size(); i++) {
        string _f_name = "Agent/agent.txt";
        string num = std::to_string(i);
        _f_name.insert(11, num);
        individual::save_indi(agent_array[i].first->root, _f_name);
    }
}

// Todo: restart strategy
void rl::rl_op() {
    auto env = CartPoleSwingUp();

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

    // size
    double size_total;

    // fitness
    double fit_total;
    double fit_utnbi;
    double fit_lgar; //last generation average reward
    int indi_best;
    individual* indi_utnb; // up to now best

    // similarity between individauls
    double sim_total;

    // rank
    vector<double> rank;

    // record the evolution process
    std::array<double, MAX_GENERATION> f_b {};
    std::array<double, MAX_GENERATION> f_a {};
    std::array<double, MAX_GENERATION> sim_a {};
    std::array<double, MAX_GENERATION> siz_a {};
    std::array<double, MAX_GENERATION/20> f_ens {};

    indi_utnb = nullptr;
    fit_utnbi = 0;
    fit_lgar = -1e6;

    // emsemble learning
    pri_que agent;

    // parallel design
    tf::Executor executor;
    tf::Taskflow taskflow;
    auto observer = executor.make_observer<tf::ExecutorObserver>();
    std::ofstream ofs("timestamps.json");

    env.reset_ini();
    agent_push(agent, pop[0], -1e6);

    //Evolution
    for (int gen = 0; gen < MAX_GENERATION; gen++) {
        env.reset_ini();
        auto new_pop = new individual* [POP_SIZE];

        indi_best = 0; fit_total = 0; sim_total = 0; size_total = 0;

        std::array<double, POP_SIZE> fit{};
        std::array<double, POP_SIZE> sim{};

        vector<agent_pair> agent_array;
        agent_flat(agent, agent_array);

        // single
//        for (int i = 0; i < POP_SIZE; i++)
//            evaluate(env, pop[i], agent_array, fit[i], sim[i]);

        // parallel
        taskflow.clear();
        for (int i = 0; i < POP_SIZE; i++) {
            auto item = taskflow.emplace([&fit, &sim, env, pop, &agent_array, i]() {
                evaluate(env, pop[i], agent_array, fit[i], sim[i]);
            });
            item.name(std::to_string(i));
        }
        executor.run(taskflow);
        executor.wait_for_all();
        observer->dump(ofs);

        for (int i = 0; i < POP_SIZE; i++) {
            sim_total += sim[i];
            fit_total += fit[i];
            size_total += pop[i]->root->size;
            indi_best = (fit[indi_best] > fit[i]) ? indi_best : i;
            agent_push(agent, pop[i], fit[i]);
        }

        f_a[gen] = fit_total / double(POP_SIZE);
        sim_a[gen] = sim_total / double(POP_SIZE);
        siz_a[gen] = size_total / double(POP_SIZE);

        if (indi_utnb == nullptr) {
            indi_utnb = new individual(*pop[indi_best]);
            fit_utnbi = fit[indi_best];
        }else{
            if (fit[indi_best] >= fit_utnbi) {
                individual::indi_clean(indi_utnb);
                indi_utnb = new individual(*pop[indi_best]);
                fit_utnbi = fit[indi_best];
            }
        }
        f_b[gen] = fit_utnbi;

        // rank mode
        double fit_rate, sim_rate;

        // original method
//        fit_rate = 1; sim_rate = 0;

        // improved method
        if (f_a[gen] >= fit_lgar) {
            fit_rate = 1; sim_rate =0;
        } else {
            if (fit[indi_best] > agent.top().second) {
                fit_rate = 0.7; sim_rate = 0.3;
            } else {
                fit_rate = 0.5; sim_rate = 0.5;
            }
        }
        fit_lgar = f_a[gen];

        // get rank
        rank.clear();
        get_rank(rank, fit, sim, fit_rate, sim_rate);

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
            individual::indi_clean(parent2);
        }

        swap(pop, new_pop);

        // free pointer;
        for (int i = 0; i < POP_SIZE; i++)
            individual::indi_clean(new_pop[i]);
        delete [] new_pop;

        // record
        spdlog::info("Gen: {:<4d} f_b: {:<8.3f} f_a: {:<8.3f} s_a: {:<6.1f} "
                     "r_f: {:<3.1f} r_s: {:<3.1f} "
                     "f_utnb: {:<8.3f} f_ens_min: {:<8.3f}", gen, fit[indi_best], f_a[gen], siz_a[gen],
                     fit_rate, sim_rate, f_b[gen], agent.top().second);

        if ((gen+1) % 50 == 0) model_save(indi_utnb, agent);
    }

    // print the average fit and dist
    ofstream _file("Result.txt");
    for (int i = 0; i < MAX_GENERATION; i++)
        _file << i << " " << f_b[i] << " " << f_ens[i/20] << " " << f_a[i] << " " << siz_a[i] << endl;
    _file.close();

    // print
    py::object _b_i = py::cast(f_b);
    py::object _f_ens = py::cast(f_ens);
    py::object _f_a = py::cast(f_a);
    py::object _siz_a = py::cast(siz_a);

    py::object plt = py::module::import("matplotlib.pyplot");
    plt.attr("figure")();
    plt.attr("subplot")(411);
    plt.attr("plot")(_b_i);
    plt.attr("subplot")(412);
    plt.attr("plot")(_f_ens);
    plt.attr("subplot")(413);
    plt.attr("plot")(_f_a);
    plt.attr("subplot")(414);
    plt.attr("plot")(_siz_a);
    plt.attr("yscale")("log");
    plt.attr("show")();

    for (int i = 0; i < POP_SIZE; i++)
        individual::indi_clean(pop[i]);
    delete [] pop;
}
