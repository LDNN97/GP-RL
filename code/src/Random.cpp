//
// Created by LDNN97 on 2020/3/1.
//
# include "../include/Parameter.h"
# include "../include/Random.h"

using namespace std;

mt19937 mt(RAND_SEED);
uniform_real_distribution<double> dis(0, 1);

int rand_int(int lower, int upper) {
    return lower + int(dis(mt) * double(upper - lower));
}

double rand_real(double lower, double upper) {
    return lower + dis(mt) * (upper - lower);
}