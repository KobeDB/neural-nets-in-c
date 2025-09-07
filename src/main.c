// .h
#include "base/md.h"
#include "base/md_alias.h"
#include "autograd/autograd.h"
#include <stdio.h>

// .c
#include "base/md.c"
#include "autograd/autograd.c"

int main(void) {
    run_neural_network_things();
}
