#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>

typedef struct {
    int is_seeded;
    uint64_t state;
} LCG;

#define LCG_A 6364136223846793005ULL
#define LCG_C 1ULL

static inline void lcg_seed(LCG *rng, uint64_t seed) {
    rng->is_seeded = 1;
    rng->state = seed;
}

static inline uint32_t lcg_next_u32(LCG *rng) {
    rng->state = rng->state * LCG_A + LCG_C;
    return (uint32_t)(rng->state >> 32);
}

static inline double lcg_next_f64(LCG *rng) {
    return lcg_next_u32(rng) / (double)(UINT32_MAX + 1.0);
}

static inline double lcg_next_range_f64(LCG *rng, double min, double max) {
    double unit = lcg_next_f64(rng); // in range [0.0, 1.0)
    return min + unit * (max - min);
}

#endif
