// .h
#include "base/md.h"
#include "base/md_alias.h"
#include "testing/testing.h"
#include "autograd/autograd.h"
#include "nn/nn_inc.h"

// .c
#include "base/md.c"
#include "testing/testing.c"
#include "autograd/autograd.c"
#include "nn/nn_inc.c"

// test functions includes
#include "test_autograd.c"
#include "test_nn.c"


int main(void) {
    Arena *arena = arena_alloc();

    T_TestResultList all_results = {0};

    //
    // Run all tests
    //
    T_RunTest(arena, &all_results, test_autograd);
    T_RunTest(arena, &all_results, test_nn);

    t_print_test_report(&all_results);

    return 0;
}
