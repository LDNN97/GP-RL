//
// Created by LDNN97 on 2020/3/1.
//

#ifndef GP_CPP_RANDOM_H
#define GP_CPP_RANDOM_H

#include <random>

using namespace std;

mt19937 mt(0); // seed()
uniform_real_distribution<double> dis(0, 1);

int rand_int(int lower, int upper) {
    return lower + int(dis(mt) * double(upper - lower));
}

#endif //GP_CPP_RANDOM_H
