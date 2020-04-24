//
// Created by LDNN97 on 2020/4/2.
//
#include "../include/RL_OP.h"
#include "spdlog/spdlog.h"

namespace py = pybind11;
using namespace rl;
using std::ofstream;
using std::ifstream;
using std::string;
using std::vector;

void rl::env_reset(py::object &env, state_arr &st){
    py::object st_or = env.attr("reset")();
    st = st_or.cast<state_arr>();
}

void rl::env_step(pybind11::object &env, const int &act, state_arr &nst, double &reward, bool &end) {
    py::list nst_or = env.attr("step")(action_set[act]);
    nst = py::cast<state_arr>(nst_or[0]);
    reward = py::cast<double>(nst_or[1]);
    end = py::cast<bool>(nst_or[2]);
}

int rl::get_max_action(py::object &env, individual* indi){
    state_arr nst{}; double reward = 0; bool end = false;
    double max_v = -1e6; int max_act = 0; double v;

    for (int i = 0; i < n_action; i++){
        rl::env_step(env, i, nst, reward, end);
        v = reward + 1 * individual::calculate(indi->root, nst);
        if (v > max_v) {
            max_v = v;
            max_act = i;
        }
        env.attr("back_step")();
    }

    return max_act;
}

int rl::ensemble_selection(py::object &env, vector<individual*> &agent) {
    std::array<int, n_action> box{};

    for (auto indi:agent) {
        state_arr nst{}; double reward = 0; bool end = false;
        double max_v = -1e6; int max_act = 0; double v;
        for (int i = 0; i < n_action; i++){
            rl::env_step(env, i, nst, reward, end);
            v = reward + 1 * individual::calculate(indi->root, nst);
            if (v > max_v) {
                max_v = v;
                max_act = i;
            }
            env.attr("back_step")();
        }
        box[max_act]++;
    }

    int ans = std::distance(box.begin(), std::max_element(box.begin(), box.end()));

    return ans;
}

void rl::best_agent(int ID) {
    pybind11::scoped_interpreter guard{};
    pybind11::module::import("sys").attr("argv").attr("append")("");
    pybind11::module::import("sys").attr("path").attr("insert")(0, "..");

    py::object env_list = py::module::import("env_py");
    py::object env = env_list.attr(env_name.c_str())();

    std::string _pre = "EXP ";
    std::string num = std::to_string(ID); // set 0 to EXP_NUM
    _pre.insert(4, num);
    _pre += " ";

    individual* indi = individual::load_indi("Agent/" + _pre + "best_agent.txt");

    std::array<double, n_observation> st{}, nst{};
    int action; double reward; bool end;

    for (int i = 0; i < 10; i++) {
        env.attr("reset_ini")();
        rl::env_reset(env, st);
        double reward_indi = 0;
        for (int step = 0; step < 1000; step++){
            action = rl::get_max_action(env, indi);
            rl::env_step(env, action, nst, reward, end);
            std::swap(st, nst);
            if (end) break;
            reward_indi += reward; //env CartPole +1 MountainCar -1
            env.attr("render")();
        }
        spdlog::info("Reward:{:>8.3f}", reward_indi);
    }
    env.attr("close")();

    individual::indi_clean(indi);
}

void rl::ensemble_agent(int ID, int ind) {
    pybind11::scoped_interpreter guard{};
    pybind11::module::import("sys").attr("argv").attr("append")("");

    py::object env_list = py::module::import("env");
    py::object env = env_list.attr(env_name.c_str())();

    std::string _pre = "EXP ";
    std::string num = std::to_string(ID);
    _pre.insert(4, num);
    _pre += " ";

    int ensemble_size = 0;
    ifstream _file("Agent/" + _pre + "ensemble_size.txt");
    _file >> ensemble_size;
    _file.close();

    vector<individual*> agent;
    if (ind == -1) {
        for (int i = 0; i < ensemble_size; i++) {
            string _f_name = "Agent/" + _pre + "agent.txt";
            string _f_num = std::to_string(i);
            _f_name.insert(_f_name.length() - 4, _f_num);
            individual* indi = individual::load_indi(_f_name);
            agent.emplace_back(indi);
        }
    }else {
        string _f_name = "Agent/" + _pre + "agent.txt";
        string _f_num = std::to_string(ind);
        _f_name.insert(_f_name.length() - 4, _f_num);
        individual* indi = individual::load_indi(_f_name);
        agent.emplace_back(indi);
    }

    std::array<double, n_observation> st{}, nst{};
    double reward; bool end;

    for (int i = 0; i < 10; i++) {
        env.attr("reset_ini")();
        std::array<int, 1000> rec_a{};
        rl::env_reset(env, st);
        double reward_indi = 0;
        for (int step = 0; step < 1000; step++){
            rec_a[step] = ensemble_selection(env, agent);
            rl::env_step(env, rec_a[step], nst, reward, end);
            std::swap(st, nst);
            if (end) break;
            reward_indi += reward;
        }
        spdlog::info("Reward:{:>8.3f}", reward_indi);

        rl::env_reset(env, st);
        for (int step = 0; step < 1000; step++){
            rl::env_step(env, rec_a[step], nst, reward, end);
            std::swap(st, nst);
            if (end) break;
            env.attr("render")();
        }
    }
    env.attr("close")();

    for (auto indi:agent)
        individual::indi_clean(indi);
    agent.clear();
}

void rl::population_agent(std::vector<int> exp_set) {
    pybind11::scoped_interpreter guard{};
    pybind11::module::import("sys").attr("argv").attr("append")("");

    py::object env_list = py::module::import("env");
    py::object env = env_list.attr(env_name.c_str())();

    vector<individual*> agent;
    for (auto i : exp_set) {
        std::string _pre = "EXP ";
        std::string num = std::to_string(i); // set 0 to EXP_NUM
        _pre.insert(4, num);
        _pre += " ";

        for (int j = 0; j < ENSEMBLE_SIZE; j++) {
            string _f_name = "Agent/" + _pre + "agent.txt";
            std::string _f_num = std::to_string(j);
            _f_name.insert(_f_name.length() - 4, _f_num);
            individual* indi = individual::load_indi(_f_name);
            agent.emplace_back(indi);
        }
    }

    std::array<double, n_observation> st{}, nst{};
    int action; double reward; bool end;

    for (int i = 0; i < 10; i++) {
        env.attr("reset_ini")();
        std::array<int, 1000> rec_a{};
        rl::env_reset(env, st);
        double reward_indi = 0;
        for (int step = 0; step < 1000; step++){
            rec_a[step] = ensemble_selection(env, agent);
            rl::env_step(env, rec_a[step], nst, reward, end);
            std::swap(st, nst);
            if (end) break;
            reward_indi += reward;
        }
        spdlog::info("Reward:{:>8.3f}", reward_indi);

        rl::env_reset(env, st);
        for (int step = 0; step < 1000; step++){
            rl::env_step(env, rec_a[step], nst, reward, end);
            std::swap(st, nst);
            if (end) break;
            env.attr("render")();
        }
    }
    env.attr("close")();

    for (auto indi:agent)
        individual::indi_clean(indi);
    agent.clear();
}

