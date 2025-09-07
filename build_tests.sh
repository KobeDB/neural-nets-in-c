#!/bin/bash
set -e

mkdir -p bin

pushd bin
clang -std=c99 -pedantic -D_GNU_SOURCE \
    ../src/tests/tests_main.c -o tests_main -g \
    -I../src \
    -lm
popd
